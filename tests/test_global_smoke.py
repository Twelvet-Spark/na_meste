"""Isolated body-adapter contracts using generated tiny ONNX models, not OSNet proof.

Run: .venv/Scripts/python.exe tests/test_global_smoke.py
No production configuration, database, weights or source photographs are loaded.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import traceback
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.ai.model_bundle import ModelBundle, ModelBundleError
from app.ai.reid import PersonReID

# Reuse the existing isolated manifest fixture rather than inventing another
# provenance convention. Importing this module only registers its standalone cases.
spec = importlib.util.spec_from_file_location(
    "body_bundle_test_support", ROOT / "tests" / "test_model_bundle.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)

TESTS = []
SAMPLE_INDICES = (np.arange(512, dtype=np.int64) * 193) % (3 * 256 * 128)


def test(fn):
    TESTS.append(fn)
    return fn


def body_graph(*, mode="normal", input_shape=None, input_type=TensorProto.FLOAT,
               output_shape=None, output_type=TensorProto.FLOAT, extra_input=False,
               extra_output=False, guard_size=17):
    """Observe RGB pixels and actual chunk size through genuine ORT computation.

    The 17-entry lookup rejects a batch greater than 16. Its small additive
    batch-size term also exposes padding a short chunk with invented crops.
    """
    initializers = [
        numpy_helper.from_array(SAMPLE_INDICES, "sample_indices"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), "batch_axis"),
        numpy_helper.from_array(np.arange(guard_size, dtype=np.float32) / 100, "batch_terms"),
    ]
    nodes = [
        helper.make_node("Cast", ["images"], ["float_images"], to=TensorProto.FLOAT),
        helper.make_node("Flatten", ["float_images"], ["pixels"], axis=1),
        helper.make_node("Gather", ["pixels", "sample_indices"], ["features"], axis=1),
        helper.make_node("Shape", ["images"], ["image_shape"]),
        helper.make_node("Gather", ["image_shape", "batch_axis"], ["batch_size"], axis=0),
        helper.make_node("Gather", ["batch_terms", "batch_size"], ["batch_term"], axis=0),
        helper.make_node("Add", ["features", "batch_term"], ["sized_features"]),
    ]
    value = "sized_features"
    if mode in ("zero", "nan", "inf", "huge"):
        number = {"zero": 0., "nan": np.nan, "inf": np.inf,
                  "huge": float(np.finfo(np.float32).max)}[mode]
        initializers.append(numpy_helper.from_array(np.array(number, dtype=np.float32), "factor"))
        if mode == "huge":
            nodes.append(helper.make_node("Shape", [value], ["feature_shape"]))
            nodes.append(helper.make_node("Expand", ["factor", "feature_shape"], ["changed_features"]))
        else:
            nodes.append(helper.make_node("Mul", [value, "factor"], ["changed_features"]))
        value = "changed_features"
    elif mode == "missing_row":
        initializers.extend([
            numpy_helper.from_array(np.array(1, dtype=np.int64), "one"),
            numpy_helper.from_array(np.array([0], dtype=np.int64), "axis_zero"),
            numpy_helper.from_array(np.array([0], dtype=np.int64), "start"),
        ])
        nodes.extend([
            helper.make_node("Sub", ["batch_size", "one"], ["short_size"]),
            helper.make_node("Unsqueeze", ["short_size", "axis_zero"], ["end"]),
            helper.make_node("Slice", [value, "start", "end", "axis_zero"], ["short_features"]),
        ])
        value = "short_features"
    elif mode == "rank_one":
        nodes.append(helper.make_node("ReduceMean", [value], ["flat_features"], axes=[0], keepdims=0))
        value = "flat_features"
    elif mode == "width_511":
        initializers[0].CopyFrom(numpy_helper.from_array(SAMPLE_INDICES[:511], "sample_indices"))
    nodes.append(helper.make_node("Cast", [value], ["embeddings"], to=output_type))
    inputs = [helper.make_tensor_value_info(
        "images", input_type, ["N", 3, 256, 128] if input_shape is None else input_shape)]
    outputs = [helper.make_tensor_value_info(
        "embeddings", output_type, ["N", 512] if output_shape is None else output_shape)]
    if extra_input:
        inputs.append(helper.make_tensor_value_info("unexpected", TensorProto.FLOAT, ["N", 1]))
    if extra_output:
        nodes.append(helper.make_node("Identity", ["embeddings"], ["unexpected"]))
        outputs.append(helper.make_tensor_value_info("unexpected", output_type, ["N", 512]))
    model = helper.make_model(
        helper.make_graph(nodes, "body-preprocessing-probe", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 16)], ir_version=10)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def fixture_at(root, **graph_options):
    fixture = support.BundleFixture(root)
    replace_body(fixture, body_graph(**graph_options))
    fixture.save()
    return fixture


def replace_body(fixture, data):
    fixture.asset("body").write_bytes(data)
    fixture.file("body")["sha256"] = hashlib.sha256(data).hexdigest()


def adapter(fixture):
    fixture.save()
    return PersonReID(ModelBundle.load(fixture.path), ["CPUExecutionProvider"])


def expect_body_error(operation, code="MODEL_CONTRACT_INVALID"):
    try:
        operation()
    except ModelBundleError as exc:
        assert exc.code == code, (exc.code, code)
        assert exc.component == "body", exc.component
        return
    raise AssertionError(f"Expected {code} [body]")


def crops(count):
    rows = []
    for index in range(count):
        height, width = ((37, 91), (113, 29), (256, 128), (17, 13))[index % 4]
        y, x = np.indices((height, width))
        rows.append(np.stack(((x * 19 + y * 3 + index * 11) % 256,
                              (x * 5 + y * 23 + index * 17) % 256,
                              (x * 13 + y * 7 + index * 29) % 256), axis=2).astype(np.uint8))
    return rows


def reference(crop, chunk_size):
    # Independent reference follows the preparer's PIL RGB/bilinear path, not a
    # production helper. High-frequency fixtures distinguish this from cv2.resize.
    rgb = Image.fromarray(crop[:, :, ::-1]).resize((128, 256), Image.Resampling.BILINEAR)
    value = np.asarray(rgb, dtype=np.float32) / 255
    value = (value - np.array([.485, .456, .406], dtype=np.float32)) / np.array(
        [.229, .224, .225], dtype=np.float32)
    selected = value.transpose(2, 0, 1).reshape(-1)[SAMPLE_INDICES] + np.float32(chunk_size / 100)
    return (selected / np.linalg.norm(selected.astype(np.float64))).astype(np.float32)


@test
def first_live_crop_does_not_charge_cuda_startup_to_appearance_budget():
    # CUDA initialization is thread-local: warming the construction thread does
    # not protect the inference owner's first live appearance deadline.
    from app.ai.worker import AIWorker
    elapsed = [0.]
    initialized = set()
    native_run = ort.InferenceSession.run
    live_costs = []
    def timed_run(session, *args, **kwargs):
        key = (id(session), threading.get_ident())
        elapsed[0] += .009 if key in initialized else .835
        initialized.add(key)
        return native_run(session, *args, **kwargs)
    class ProbeWorker(AIWorker):
        def step(self):
            started = elapsed[0]
            vectors = self.reid.embed(crops(1))
            live_costs.append((elapsed[0] - started, vectors.shape))
            self._stop_event.set()
            return True
        def flush(self, reason):
            pass
    with tempfile.TemporaryDirectory() as directory, patch.object(ort.InferenceSession, "run", timed_run):
        reid = adapter(fixture_at(Path(directory)))
        worker = ProbeWorker(None, SimpleNamespace(store=None), None, None, reid=reid)
        worker.start()
        worker.join()
        assert len(live_costs) == 1
        cost, shape = live_costs[0]
        assert cost < .02, 'cold-start cost must not permanently exhaust the live appearance budget'
        assert shape == (1, 512)


@test
def empty_input_returns_typed_empty_embeddings_without_native_inference():
    with tempfile.TemporaryDirectory() as directory:
        reid = adapter(fixture_at(Path(directory)))
        assert isinstance(reid.session, ort.InferenceSession)
        assert reid.session.get_providers() == ["CPUExecutionProvider"]
        with patch.object(reid.session, "run", side_effect=AssertionError("empty inference")):
            output = reid.embed([])
        assert output.shape == (0, 512) and output.dtype == np.float32


@test
def one_crop_preserves_rgb_order_and_imagenet_normalization():
    with tempfile.TemporaryDirectory() as directory:
        reid = adapter(fixture_at(Path(directory)))
        crop = np.full((27, 63, 3), [17, 83, 239], dtype=np.uint8)
        result = reid.embed([crop])
        assert result.shape == (1, 512) and result.dtype == np.float32
        # Hand-derived channel values, sampled by the ONNX graph across CHW.
        channels = np.array([(239 / 255 - .485) / .229,
                             (83 / 255 - .456) / .224,
                             (17 / 255 - .406) / .225], dtype=np.float64)
        expected = channels[SAMPLE_INDICES // (256 * 128)] + .01
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result[0], expected, rtol=2e-6, atol=2e-7)


@test
def four_different_aspects_use_pil_bilinear_without_mutating_crops():
    with tempfile.TemporaryDirectory() as directory:
        reid = adapter(fixture_at(Path(directory)))
        images = crops(4)
        images[1] = images[1][::-1, ::-1]  # Native reader slices need not be contiguous.
        saved = [image.copy() for image in images]
        result = reid.embed(images)
        assert result.shape == (4, 512)
        np.testing.assert_allclose(result, np.stack([reference(image, 4) for image in images]),
                                   rtol=3e-6, atol=2e-7)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=2e-6)
        for before, after in zip(saved, images):
            np.testing.assert_array_equal(before, after)


@test
def seventeen_crops_restore_all_ordered_rows_with_no_padding_or_truncation():
    with tempfile.TemporaryDirectory() as directory:
        reid = adapter(fixture_at(Path(directory)))
        images = crops(17)
        result = reid.embed(images)
        assert result.shape == (17, 512) and result.dtype == np.float32
        expected = np.stack([reference(image, 16 if index < 16 else 1)
                             for index, image in enumerate(images)])
        np.testing.assert_allclose(result, expected, rtol=3e-6, atol=2e-7)
        assert len({row.tobytes() for row in result}) == 17
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=2e-6)


@test
def missing_or_hash_invalid_body_fails_before_session_creation():
    with tempfile.TemporaryDirectory() as directory:
        fixture = fixture_at(Path(directory))
        fixture.asset("body").write_bytes(b"not the declared model")
        mismatched = ModelBundle.load(fixture.path)
        fixture.asset("body").unlink()
        missing = ModelBundle.load(fixture.path)
        with patch("app.ai.reid.ort.InferenceSession", side_effect=AssertionError("ORT must not start")):
            expect_body_error(lambda: PersonReID(missing, ["CPUExecutionProvider"]), "MODEL_MISSING")
            expect_body_error(lambda: PersonReID(mismatched, ["CPUExecutionProvider"]),
                              "MODEL_HASH_MISMATCH")


@test
def actual_session_metadata_mismatches_fail_closed():
    variants = [
        {"input_shape": [1, 3, 256, 128]},
        {"input_shape": ["N", 1, 256, 128]},
        {"input_shape": ["N", 3, 112, 128]},
        {"input_shape": ["N", 3, 256, 256]},
        {"input_type": TensorProto.FLOAT16},
        {"mode": "width_511", "output_shape": ["N", 511]},
        {"mode": "rank_one", "output_shape": [512]},
        {"output_type": TensorProto.FLOAT16},
        {"extra_input": True},
        {"extra_output": True},
    ]
    with tempfile.TemporaryDirectory() as directory:
        fixture = fixture_at(Path(directory))
        bundle = ModelBundle.load(fixture.path)
        variants = [
            {"input_shape": [1, 3, 256, 128]},
            {"input_shape": ["N", 1, 256, 128]},
            {"input_shape": ["N", 3, 112, 128]},
            {"input_shape": ["N", 3, 256, 256]},
            {"input_shape": ["N", 3, 256]},
            {"input_type": TensorProto.FLOAT16},
            {"mode": "rank_one", "output_shape": [512]},
            {"mode": "width_511", "output_shape": ["N", 511]},
            {"output_type": TensorProto.FLOAT16},
            {"extra_input": True},
            {"extra_output": True},
        ]
        for options in variants:
            # Replace a validated asset afterwards so this checks the adapter's
            # actual ORT metadata gate, not just ModelBundle's manifest validator.
            fixture.asset("body").write_bytes(body_graph(**options))
            expect_body_error(lambda: PersonReID(bundle, ["CPUExecutionProvider"]))


@test
def actual_inference_cannot_silently_drop_a_row():
    with tempfile.TemporaryDirectory() as directory:
        fixture = fixture_at(Path(directory), mode="missing_row")
        expect_body_error(lambda: adapter(fixture).embed(crops(4)))


@test
def actual_nonfinite_and_zero_norm_outputs_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        fixture = fixture_at(Path(directory))
        for mode in ("nan", "inf", "zero"):
            replace_body(fixture, body_graph(mode=mode))
            expect_body_error(lambda: adapter(fixture).embed(crops(1)))


@test
def finite_large_outputs_normalize_without_float32_norm_overflow():
    with tempfile.TemporaryDirectory() as directory:
        reid = adapter(fixture_at(Path(directory), mode="huge"))
        result = reid.embed(crops(1))
        np.testing.assert_allclose(result, np.full((1, 512), 1 / np.sqrt(512)), atol=2e-7)
        assert np.isfinite(result).all()


@test
def actual_native_inference_errors_use_the_static_body_error():
    with tempfile.TemporaryDirectory() as directory:
        reid = adapter(fixture_at(Path(directory), guard_size=2))
        expect_body_error(lambda: reid.embed(crops(4)))


@test
def malformed_or_nonfinite_crops_fail_closed():
    with tempfile.TemporaryDirectory() as directory:
        reid = adapter(fixture_at(Path(directory)))
        invalid = [None, np.zeros((3, 4), dtype=np.uint8),
                   np.zeros((3, 4, 4), dtype=np.uint8), np.zeros((0, 4, 3), dtype=np.uint8),
                   np.zeros((3, 0, 3), dtype=np.uint8),
                   np.full((3, 4, 3), np.nan, dtype=np.float32),
                   np.full((3, 4, 3), np.inf, dtype=np.float32)]
        for crop in invalid:
            expect_body_error(lambda: reid.embed([crop]))


@test
def body_model_id_tracks_declared_body_bytes_not_paths_face_or_scene():
    with tempfile.TemporaryDirectory() as directory:
        fixture = fixture_at(Path(directory))
        original = adapter(fixture).model_id
        body_hash = fixture.file("body")["sha256"]
        declaration = {"preprocessing_id": "osnet-rgb256x128-imagenet-v1",
                       "files": {"model": body_hash}}
        expected = hashlib.sha256(json.dumps(
            declaration, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
        assert original == expected and len(original) == 64
        fixture.replace_graph("face_recognition", doc="changed face model, same body")
        fixture.replace_graph("scene_encoder", doc="changed scene model, same body")
        old_path = fixture.asset("body")
        fixture.file("body")["path"] = "assets/renamed-body.onnx"
        old_path.rename(fixture.asset("body"))
        assert adapter(fixture).model_id == original
        model = onnx.load_model(fixture.asset("body"))
        model.doc_string = "different declared body artifact"
        replace_body(fixture, model.SerializeToString())
        assert adapter(fixture).model_id != original


if __name__ == "__main__":
    failed = 0
    for case in TESTS:
        try:
            case()
            print("PASS", case.__name__)
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(bool(failed))
