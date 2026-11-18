"""Isolated biometric provenance/migration and adapter regressions.

Run: .venv/Scripts/python.exe tests/test_gallery_migration.py
Only ONNX execution is doubled; SQLite, image validation and alignment are real.
The integration owner separately runs the approved real-model parity/smoke proof.
"""
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.ai.face_detector import FaceEngine, FaceInfo, norm_crop
from app.ai.face_recognition import EmbeddingStore
from app.ai.model_bundle import ModelBundleError
from app.database.database import immediate_session
from app.database.models import AttendanceRuntime, Employee, EmployeeFace, FaceModelMetadata
from app.services.gallery_migration import GalleryChangedError, rebuild_gallery, reembed_aligned_thumbnails
from test_operator_storage import committed_v3, database, dump, schema

MODEL = "a" * 64
OTHER_MODEL = "b" * 64
ALIGNMENT = "arcface-5pt-112-v1"
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def vector(index=0):
    result = np.zeros(512, np.float32)
    result[index] = 1.
    return result


def thumbnail(value=0, *, extension=".jpg", size=112):
    ok, encoded = cv2.imencode(extension, np.full((size, size, 3), value, np.uint8))
    assert ok
    return encoded.tobytes()


class Recognizer:
    model_id = MODEL
    alignment_id = ALIGNMENT

    def __init__(self):
        self.batches = []
        self.hook = None
        self.output = None

    def embed_aligned(self, images):
        self.batches.append([int(image[0, 0, 0]) for image in images])
        if self.hook:
            self.hook()
        if self.output is not None:
            return self.output
        return np.stack([vector(int(image[0, 0, 0])) for image in images])


def sample(db, owner, embedding=None, thumb=None, model=None, alignment=ALIGNMENT):
    employee = db.get(Employee, owner)
    if employee is None:
        db.add(Employee(id=owner, name=f"Subject {owner}"))
        db.flush()
    face = EmployeeFace(employee_id=owner, embedding=vector().tobytes() if embedding is None else embedding,
                        thumbnail=thumb)
    db.add(face)
    db.flush()
    if model is not None:
        db.add(FaceModelMetadata(face_id=face.id, model_id=model, alignment_id=alignment))
    return face.id


@test
def test_v3_upgrade_preserves_unconvertible_samples_before_target_publication():
    with database() as (engine, _root):
        committed_v3(engine)
        with immediate_session(engine) as db:
            convertible = sample(db, 41, thumb=thumbnail(2))
            legacy = sample(db, 41, embedding=b"unconvertible legacy bytes")
        schema.initialize_attendance_schema(engine)
        report = rebuild_gallery(engine, Recognizer())
        assert report["faces_reencoded"] == 1 and report["employees_needing_photos"] == [41]
        with Session(engine) as db:
            assert db.get(EmployeeFace, convertible).embedding == vector(2).tobytes()
            assert db.get(EmployeeFace, legacy).embedding == b"unconvertible legacy bytes"
            assert db.get(FaceModelMetadata, legacy) is None
            assert db.get(Employee, 41).name == "Не разбивать старое имя"
            store = EmbeddingStore(MODEL)
            assert store.refresh(db) == 1 and store.match(vector(2)).employee_id == 41
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT version FROM operator_schema").scalar_one() == 7
            assert connection.exec_driver_sql(
                "SELECT count(*) FROM sqlite_master WHERE name='presence_sessions'").scalar_one() == 0


@test
def test_gallery_only_matches_proven_normalized_target_space_but_retains_employee_names():
    with database() as (engine, _root):
        schema.initialize_attendance_schema(engine)
        with immediate_session(engine) as db:
            sample(db, 1, model=MODEL)
            sample(db, 2, vector(1).tobytes())
            sample(db, 3, vector(2).tobytes(), model=OTHER_MODEL)
            sample(db, 4, (vector(3) * 2).tobytes(), model=MODEL)
            sample(db, 5, vector(4).tobytes(), model=MODEL, alignment="other-alignment")
            sample(db, 6, b"malformed", model=MODEL)
            sample(db, 7, vector(5).tobytes(), model=MODEL)
            db.get(Employee, 7).active = False
        store = EmbeddingStore(MODEL)
        with Session(engine) as db:
            assert store.refresh(db) == 1
        assert store.model_id == MODEL and store.match(vector()).employee_id == 1
        for owner in (2, 3, 4, 5, 6):
            assert store.employee_name(owner) == f"Subject {owner}" and store.is_active(owner)
        assert not store.is_active(7)
        assert EmbeddingStore(OTHER_MODEL).model_id == OTHER_MODEL
        assert store.employee_name(7) == "Subject 7"
        incompatible_store = EmbeddingStore(OTHER_MODEL)
        with Session(engine) as db:
            assert incompatible_store.refresh(db) == 1
        assert incompatible_store.match(vector(2)).employee_id == 3
        assert incompatible_store.match(vector(2)).score == 1.
        assert store.match(vector(2)).score == 0.


@test
def test_thumbnail_preparation_preserves_order_validates_jpeg_and_chunks_at_four():
    recognizer = Recognizer()
    images = [None, b"broken", thumbnail(3, extension=".png"), thumbnail(4, size=113)]
    images += [thumbnail(index) for index in range(17)]
    results = reembed_aligned_thumbnails(recognizer, images)
    assert len(results) == 21 and all(row is None for row in results[:4])
    for index, result in enumerate(results[4:]):
        assert np.array_equal(result, vector(index))
    assert [len(batch) for batch in recognizer.batches] == [4, 4, 4, 4, 1]


@test
def test_invalid_model_outputs_raise_instead_of_silently_retaining_legacy_vectors():
    for output in (np.ones((1, 511)), np.full((1, 512), np.nan), np.zeros((1, 512)),
                   np.ones((2, 512))):
        recognizer = Recognizer()
        recognizer.output = output
        try:
            reembed_aligned_thumbnails(recognizer, [thumbnail()])
        except ModelBundleError as exc:
            assert exc.code == "MODEL_CONTRACT_INVALID"
        else:
            raise AssertionError("Malformed recognizer output was accepted")


@test
def test_rebuild_keeps_valid_without_thumbnail_reencodes_only_needed_and_is_idempotent():
    with database() as (engine, _root):
        schema.initialize_attendance_schema(engine)
        with immediate_session(engine) as db:
            valid = sample(db, 1, model=MODEL)
            converted = sample(db, 2, thumb=thumbnail(2))
            sample(db, 2, thumb=thumbnail(7))
            incompatible = sample(db, 3, thumb=thumbnail(3), model=OTHER_MODEL)
            invalid = sample(db, 4, embedding=b"invalid", thumb=thumbnail(4), model=MODEL)
            missing = sample(db, 5)
            malformed = sample(db, 6, thumb=b"bad", model=OTHER_MODEL)
        recognizer = Recognizer()
        report = rebuild_gallery(engine, recognizer)
        assert report["faces_reencoded"] == 4 and report["usable_faces_count"] == 5
        assert report["employees_needing_photos"] == [5, 6]
        with Session(engine) as db:
            assert db.get(EmployeeFace, valid).embedding == vector().tobytes()
            for face_id, index in ((converted, 2), (incompatible, 3), (invalid, 4)):
                assert db.get(EmployeeFace, face_id).embedding == vector(index).tobytes()
                metadata = db.get(FaceModelMetadata, face_id)
                assert metadata.model_id == MODEL and metadata.alignment_id == ALIGNMENT
            assert db.get(EmployeeFace, missing).embedding == vector().tobytes()
            assert db.get(FaceModelMetadata, missing) is None
            assert db.get(FaceModelMetadata, malformed).model_id == OTHER_MODEL
            assert db.get(AttendanceRuntime, 1).gallery_revision == 1
            assert db.get(AttendanceRuntime, 1).schema_version == 3
            assert dict(db.query(Employee.id, Employee.revision).all()) == {
                1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0,
            }
        before, calls = dump(engine), len(recognizer.batches)
        restarted = rebuild_gallery(engine, recognizer)
        assert restarted["faces_reencoded"] == 0 and restarted["employees_needing_photos"] == [5, 6]
        assert dump(engine) == before and len(recognizer.batches) == calls


@test
def test_compatible_restart_never_acquires_writer():
    with database() as (engine, _root):
        schema.initialize_attendance_schema(engine)
        with immediate_session(engine) as db:
            sample(db, 1, model=MODEL)
            sample(db, 2, thumb=thumbnail(2), model=MODEL)
        before = dump(engine)
        statements = []

        def capture(connection, cursor, statement, parameters, context, executemany):
            statements.append(statement.upper())

        event.listen(engine, "before_cursor_execute", capture)
        try:
            recognizer = Recognizer()
            assert rebuild_gallery(engine, recognizer)["faces_reencoded"] == 0
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert not any(sql.startswith(("BEGIN IMMEDIATE", "UPDATE", "INSERT", "DELETE")) for sql in statements)
        assert recognizer.batches == [] and dump(engine) == before


@test
def test_concurrent_gallery_or_employee_edit_discards_all_prepared_vectors():
    for gallery_changed in (True, False):
        with database() as (engine, _root):
            schema.initialize_attendance_schema(engine)
            with immediate_session(engine) as db:
                face_id = sample(db, 1, thumb=thumbnail(1))
            recognizer = Recognizer()

            def edit():
                with immediate_session(engine) as db:
                    if gallery_changed:
                        db.get(AttendanceRuntime, 1).gallery_revision += 1
                    else:
                        db.get(Employee, 1).revision += 1
                        db.get(Employee, 1).name = "Operator edit"

            recognizer.hook = edit
            try:
                rebuild_gallery(engine, recognizer)
            except GalleryChangedError as exc:
                assert exc.code == "GALLERY_CHANGED"
            else:
                raise AssertionError("Concurrent operator change was overwritten")
            with Session(engine) as db:
                assert db.get(EmployeeFace, face_id).embedding == vector().tobytes()
                assert db.get(FaceModelMetadata, face_id) is None
                assert db.get(Employee, 1).name == ("Subject 1" if gallery_changed else "Operator edit")


@test
def test_failed_inference_or_commit_leaves_all_old_data_and_revision_intact():
    for failure in ("inference", "commit"):
        with database() as (engine, _root):
            schema.initialize_attendance_schema(engine)
            with immediate_session(engine) as db:
                sample(db, 1, thumb=thumbnail(1))
            before = dump(engine)
            recognizer = Recognizer()
            if failure == "inference":
                recognizer.hook = lambda: (_ for _ in ()).throw(RuntimeError("injected inference failure"))

            def fail_write(connection, cursor, statement, parameters, context, executemany):
                if statement.upper().startswith("UPDATE ATTENDANCE_RUNTIME"):
                    raise OperationalError(statement, parameters, RuntimeError("injected commit failure"))

            if failure == "commit":
                event.listen(engine, "before_cursor_execute", fail_write)
            try:
                try:
                    rebuild_gallery(engine, recognizer)
                except (RuntimeError, OperationalError):
                    pass
                else:
                    raise AssertionError("Injected failure was swallowed")
            finally:
                if failure == "commit":
                    event.remove(engine, "before_cursor_execute", fail_write)
            assert dump(engine) == before


class Bundle:
    face_model_id = MODEL
    alignment_id = ALIGNMENT

    def require(self, component):
        return Path(component + ".onnx")


class AdapterSession:
    """Named ONNX tensor boundary; no model downloads or runtime inference."""
    def __init__(self, component):
        self.component = component
        self.feeds = []
        self.bad_output = False

        self.head_transform = None
    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        names = [f"{head}_{stride}" for head in ("cls", "obj", "bbox", "kps") for stride in (8, 16, 32)]
        return [SimpleNamespace(name=name) for name in (names if self.component == "face_detection" else ["embedding"])]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, feed):
        tensor = feed["input"]
        self.feeds.append(tensor.copy())
        if self.component == "face_recognition":
            output = np.zeros((len(tensor), 512), np.float32)
            output[:, 0] = 1.
            output[:, 1] = tensor[:, 0, 0, 0]
            return [output[:1] if self.bad_output else output]
        height, width = tensor.shape[2:]
        outputs = {}
        for stride in (8, 16, 32):
            count = (height // stride) * (width // stride)
            for head, channels in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)):
                outputs[f"{head}_{stride}"] = np.zeros((1, count, channels), np.float32)
        index = width // 8 + 3
        outputs["cls_8"][0, index, 0] = 1.21
        outputs["obj_8"][0, index, 0] = .81
        outputs["bbox_8"][0, index] = [1., 1., np.log(2), np.log(2)]
        outputs["kps_8"][0, index] = [.5, .5, 1.5, .5, 1., 1., .5, 1.5, 1.5, 1.5]
        if self.head_transform is not None:
            self.head_transform(outputs)
        return [outputs[name] for name in names]


def adapter():
    detection, recognition = AdapterSession("face_detection"), AdapterSession("face_recognition")
    sessions = {"face_detection.onnx": detection, "face_recognition.onnx": recognition}
    with patch("onnxruntime.InferenceSession", side_effect=lambda path, **kw: sessions[str(path)]):
        engine = FaceEngine(Bundle(), ["CUDAExecutionProvider", "CPUExecutionProvider"], det_size=64)
    return engine, detection, recognition


@test
def test_yunet_raw_bgr_heads_native_coordinates_and_detection_only_contract():
    engine, detection, recognition = adapter()
    image = np.full((40, 80, 3), [10, 30, 90], np.uint8)
    faces = engine.detect_faces(image)
    assert len(faces) == 1 and recognition.feeds == []
    detected = faces[0]
    assert np.allclose(detected.bbox, [30., 10., 50., 30.], atol=1e-5)
    assert np.array_equal(detected.native_bbox, detected.bbox)
    assert np.allclose(detected.landmarks[0], [35., 15.])
    assert abs(detected.det_score - .9) < 1e-6
    assert detected.model_id == MODEL and detected.embedding is None
    assert detected.aligned.shape == (112, 112, 3)
    assert np.array_equal(detection.feeds[0][0, :, 0, 0], [10, 30, 90])
    assert engine.providers() == {"detection": ["CPUExecutionProvider"], "recognition": ["CPUExecutionProvider"]}


@test
def test_arcface_alignment_preserves_template_coordinates_and_old_face_positionals():
    points = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                       [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
    image = np.arange(112 * 112 * 3, dtype=np.uint8).reshape(112, 112, 3)
    assert np.array_equal(norm_crop(image, points), image)
    original = FaceInfo(np.array([0, 0, 80, 80]), .99, vector(), image)
    assert original.aligned is image and original.embedding[0] == 1


@test
def test_bad_landmarks_keep_face_metadata_without_embedding_unaligned_pixels():
    engine, detection, _recognition = adapter()
    detection.head_transform = lambda heads: heads["kps_8"].fill(0)
    faces = engine.detect(np.full((40, 80, 3), 90, dtype=np.uint8))
    assert len(faces) == 1 and faces[0].model_id == MODEL
    assert faces[0].aligned is None and faces[0].embedding is None
    assert np.allclose(faces[0].bbox, [30., 10., 50., 30.])


@test
def test_embed_all_seventeen_rgb_normalized_order_and_reject_truncation():
    engine, _detection, recognition = adapter()
    crops = [np.full((112, 112, 3), [4, 20, index], np.uint8) for index in range(17)]
    embeddings = engine.embed_aligned(crops)
    assert embeddings.shape == (17, 512)
    assert [len(batch) for batch in recognition.feeds] == [16, 1]
    for index, row in enumerate(embeddings):
        expected = np.array([1., (index - 127.5) / 127.5])
        expected /= np.linalg.norm(expected)
        assert np.allclose(row[:2], expected, atol=1e-6)
        assert np.isclose(np.linalg.norm(row), 1.)
    assert engine.embed_aligned([]).shape == (0, 512)
    recognition.bad_output = True
    try:
        engine.embed_aligned(crops)
    except ModelBundleError as exc:
        assert exc.code == "MODEL_CONTRACT_INVALID"
    else:
        raise AssertionError("Truncated model batch was accepted")


@test
def test_enrollment_detection_embeds_at_most_four_and_respects_maximum():
    engine, _detection, recognition = adapter()
    faces = [FaceInfo(np.array([0, 0, 80, 80]), .99, None,
                      np.full((112, 112, 3), [0, 0, index], np.uint8), model_id=MODEL) for index in range(17)]
    with patch.object(engine, "detect_faces", return_value=faces):
        results = engine.detect(np.zeros((80, 80, 3), np.uint8))
        assert len(results) == 17 and all(row.embedding is not None for row in results)
        assert [len(batch) for batch in recognition.feeds] == [4, 4, 4, 4, 1]
        recognition.feeds.clear()
        assert len(engine.detect(np.zeros((80, 80, 3), np.uint8), max_num=3)) == 3
        assert [len(batch) for batch in recognition.feeds] == [3]


@test
def test_yunet_suppresses_overlapping_heads_and_rejects_malformed_tensors():
    engine, detection, _recognition = adapter()

    def duplicate_head(outputs):
        outputs["cls_16"][0, 1] = 1.
        outputs["obj_16"][0, 1] = .7225
        outputs["bbox_16"][0, 1] = [1., 1., 0., 0.]
        outputs["kps_16"][0, 1] = [0., .5, 1., .5, .5, 1., 0., 1.5, 1., 1.5]

    detection.head_transform = duplicate_head
    assert len(engine.detect_faces(np.zeros((40, 80, 3), np.uint8))) == 1
    for invalid in ("shape", "nonfinite"):
        def corrupt(outputs):
            if invalid == "shape":
                outputs["cls_16"] = outputs["cls_16"][:, :1]
            else:
                outputs["cls_16"][0, 0, 0] = np.nan
        detection.head_transform = corrupt
        try:
            engine.detect_faces(np.zeros((40, 80, 3), np.uint8))
        except ModelBundleError as exc:
            assert exc.code == "MODEL_CONTRACT_INVALID" and exc.component == "face_detection"
        else:
            raise AssertionError("Malformed YuNet output was accepted")


@test
def test_yunet_portrait_rounding_maps_each_native_axis_without_distortion():
    engine, detection, _recognition = adapter()

    def portrait(outputs):
        for head in outputs.values():
            head.fill(0)
        index = 3 * 4 + 1
        outputs["cls_8"][0, index] = 1.
        outputs["obj_8"][0, index] = 1.
        outputs["bbox_8"][0, index] = [1., 1., np.log(2), np.log(2)]
        outputs["kps_8"][0, index] = [.5, .5, 1.5, .5, 1., 1., .5, 1.5, 1.5, 1.5]

    detection.head_transform = portrait
    faces = engine.detect_faces(np.zeros((81, 40, 3), np.uint8))
    assert len(faces) == 1
    assert np.allclose(faces[0].bbox, [10., 30.375, 30., 50.625], atol=1e-5)
    assert detection.feeds[0].shape == (1, 3, 64, 32)


@test
def test_missing_component_fails_before_constructing_any_inference_session():
    class MissingBundle(Bundle):
        def require(self, component):
            if component == "face_recognition":
                raise ModelBundleError("MODEL_MISSING", component)
            return super().require(component)

    with patch("onnxruntime.InferenceSession", side_effect=AssertionError("Session created before require")):
        try:
            FaceEngine(MissingBundle(), ["CPUExecutionProvider"])
        except ModelBundleError as exc:
            assert exc.code == "MODEL_MISSING" and exc.component == "face_recognition"
        else:
            raise AssertionError("Missing recognizer was accepted")


if __name__ == "__main__":
    failures = 0
    for case in TESTS:
        try:
            case()
            print(f"PASS {case.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {case.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failures}/{len(TESTS)} tests passed")
    sys.exit(bool(failures))
