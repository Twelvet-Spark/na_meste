"""Isolated manifest and tiny-ONNX detector contracts; no production DB or weights.

Run: .venv/Scripts/python.exe tests/test_model_bundle.py
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from unittest import SkipTest
from unittest.mock import patch

import onnx
from onnx import TensorProto, helper

SOURCE = Path(__file__).resolve().parent.parent / "app" / "ai" / "model_bundle.py"
spec = importlib.util.spec_from_file_location("model_bundle_under_test", SOURCE)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
ModelBundle = subject.ModelBundle
ModelBundleError = subject.ModelBundleError
TESTS = []

DFINE_REV = "956d1709314c2c6a4df6f34de232054578a7449f"
YUNET_REV = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
AURA_REV = "af6d057c9b0ec4071d4c49c80e3539258798b609"
BODY_REV = "a5c5cc037c24235cda3b21085b93ad77c9616224"
BODY_EXPORT_REV = "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"
SCENE_REV = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
BODY_FILE = ("osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_"
             "b64_fb10_softmax_labsmth_flip_jitter.pth")


def test(fn):
    TESTS.append(fn)
    return fn


def tensor(name, shape, dtype="float32"):
    return {"name": name, "dtype": dtype, "shape": list(shape)}


def contracts():
    heads = [tensor(f"{head}_{stride}", [1, f"cells_{stride}", channels])
             for stride in (8, 16, 32)
             for head, channels in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10))]
    return {
        "person": ([tensor("images", ["N", 3, 640, 640]),
                    tensor("orig_target_sizes", ["N", 2], "int64")],
                   [tensor("labels", ["N", 300], "int64"),
                    tensor("boxes", ["N", 300, 4]), tensor("scores", ["N", 300])]),
        "face_detection": ([tensor("input", [1, 3, "height", "width"])], heads),
        "face_recognition": ([tensor("input.1", ["N", 3, 112, 112])],
                             [tensor("683", ["N", 512])]),
        "body": ([tensor("images", ["N", 3, 256, 128])],
                 [tensor("embeddings", ["N", 512])]),
        "scene_encoder": ([tensor("images", [1, 3, 1024, 1024])],
                          [tensor("image_embeddings", [1, 256, 64, 64])]),
        "scene_decoder": (
            [tensor("image_embeddings", [1, 256, 64, 64]),
             tensor("point_coords", [1, "num_points", 2]),
             tensor("point_labels", [1, "num_points"]),
             tensor("mask_input", [1, 1, 256, 256]),
             tensor("has_mask_input", [1]), tensor("orig_im_size", [2])],
            [tensor("masks", [1, 4, "height", "width"]),
             tensor("iou_predictions", [1, 4]), tensor("low_res_masks", [1, 4, 256, 256])]),
    }


def graph_bytes(inputs, outputs, *, doc=""):
    """Small genuine ONNX graphs: represent large tensors without allocating them."""
    types = {"float32": TensorProto.FLOAT, "int64": TensorProto.INT64,
             "float16": TensorProto.FLOAT16}
    info = lambda item: helper.make_tensor_value_info(
        item["name"], types[item["dtype"]], item["shape"])
    nodes = []
    for index, output in enumerate(outputs):
        shape_name = f"fixture_shape_{index}"
        shape = [dimension if isinstance(dimension, int) else 1
                 for dimension in output["shape"]]
        nodes.append(helper.make_node(
            "Constant", [], [shape_name],
            value=helper.make_tensor(shape_name, TensorProto.INT64, [len(shape)], shape)))
        nodes.append(helper.make_node(
            "ConstantOfShape", [shape_name], [output["name"]],
            value=helper.make_tensor(f"zero_{index}", types[output["dtype"]], [1], [0])))
    graph = helper.make_graph(nodes, "bundle-fixture", [info(v) for v in inputs],
                              [info(v) for v in outputs])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.doc_string = doc
    return model.SerializeToString()


class BundleFixture:
    def __init__(self, root):
        self.root = root
        self.path = root / "manifest.json"
        self.document = {"version": 1, "alignment_id": "arcface-5pt-112-v1", "components": {}}
        self.specifications = contracts()
        provenance = {
            "person": ("dfine-rgb-letterbox640-black-div255-v1", "Apache-2.0",
                       "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
                       "release:177720969:asset:196374453", DFINE_REV,
                       f"https://github.com/Peterande/D-FINE/blob/{DFINE_REV}/LICENSE"),
            "face_detection": ("yunet-bgr-f32-strides8-16-32-v1", "MIT",
                               f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{YUNET_REV}/models/face_detection_yunet/face_detection_yunet_2026may.onnx",
                               YUNET_REV, None,
                               f"https://github.com/opencv/opencv_zoo/blob/{YUNET_REV}/models/face_detection_yunet/LICENSE"),
            "face_recognition": ("auraface-rgb112-minus127.5-div127.5-v1", "Apache-2.0",
                                 f"https://huggingface.co/fal/AuraFace-v1/resolve/{AURA_REV}/glintr100.onnx",
                                 AURA_REV, None,
                                 f"https://huggingface.co/fal/AuraFace-v1/blob/{AURA_REV}/LICENSE.md"),
            "body": ("osnet-rgb256x128-imagenet-v1", "MIT",
                     f"https://huggingface.co/kaiyangzhou/osnet/resolve/{BODY_REV}/{BODY_FILE}",
                     BODY_REV, BODY_EXPORT_REV,
                     f"https://huggingface.co/kaiyangzhou/osnet/blob/{BODY_REV}/README.md"),
            "scene": ("mobilesam-rgb-longest1024-meanstd-pad-v1", "Apache-2.0",
                      f"https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/{SCENE_REV}/weights/mobile_sam.pt",
                      SCENE_REV, SCENE_REV,
                      f"https://github.com/ChaoningZhang/MobileSAM/blob/{SCENE_REV}/LICENSE"),
        }
        for component, (preprocessing, license_id, url, revision, exporter, license_url) in provenance.items():
            files = []
            for role in (("encoder", "decoder") if component == "scene" else ("model",)):
                key = f"scene_{role}" if component == "scene" else component
                inputs, outputs = self.specifications[key]
                files.append({
                    "role": role, "path": f"assets/{key}.onnx", "sha256": "0" * 64,
                    "source_url": url, "source_revision": revision,
                    "source_sha256": "1" * 64, "license": license_id,
                    "license_url": license_url,
                    "exporter_revision": f"{exporter}:{'2' * 64}" if exporter else f"native:{revision}",
                    "inputs": copy.deepcopy(inputs), "outputs": copy.deepcopy(outputs),
                })
            self.document["components"][component] = {"preprocessing_id": preprocessing, "files": files}
        for key in self.specifications:
            self.replace_graph(key)
        self.save()

    def file(self, key):
        component, role = ("scene", key[6:]) if key.startswith("scene_") else (key, "model")
        return next(item for item in self.document["components"][component]["files"]
                    if item["role"] == role)

    def asset(self, key):
        return self.root / self.file(key)["path"]

    def replace_graph(self, key, inputs=None, outputs=None, *, doc=""):
        default_inputs, default_outputs = self.specifications[key]
        data = graph_bytes(default_inputs if inputs is None else inputs,
                           default_outputs if outputs is None else outputs, doc=doc)
        self.asset(key).parent.mkdir(parents=True, exist_ok=True)
        self.asset(key).write_bytes(data)
        self.file(key)["sha256"] = hashlib.sha256(data).hexdigest()
        if self.file(key)["exporter_revision"].startswith("native:"):
            self.file(key)["source_sha256"] = self.file(key)["sha256"]

    def save(self):
        self.path.write_text(json.dumps(self.document), encoding="utf-8")

    def load(self):
        self.save()
        return ModelBundle.load(self.path)


def expect_error(operation, code="MODEL_CONTRACT_INVALID", component=None):
    try:
        operation()
    except ModelBundleError as exc:
        assert exc.code == code, (exc.code, code)
        assert exc.component == component, (exc.component, component)
        assert code in str(exc)
        return exc
    raise AssertionError(f"Expected {code} for {component}")


def isolated(bundle, component, status):
    assert bundle.component_status == {
        key: status if key == component else "READY"
        for key in ("person", "face_detection", "face_recognition", "body", "scene")
    }, bundle.component_status
    expect_error(lambda: bundle.require(component), status, component)
    for key in bundle.component_status:
        if key != component:
            assert bundle.require(key).is_file()


@test
def native_onnx_cannot_claim_an_unrelated_source_hash():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        fixture.file("face_recognition")["source_sha256"] = "9" * 64
        expect_error(fixture.load)


@test
def converted_auraface_retains_source_and_exporter_provenance():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        file = fixture.file("face_recognition")
        file["source_sha256"] = "9" * 64
        file["exporter_revision"] = f"{AURA_REV}:{'2' * 64}"
        bundle = fixture.load()
        assert bundle.require("face_recognition") == fixture.asset("face_recognition").resolve()
        file["exporter_revision"] = f"{'8' * 40}:{'2' * 64}"
        expect_error(fixture.load)


@test
def valid_graphs_expose_exact_paths_metadata_and_scene_encoder():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        bundle = fixture.load()
        assert set(bundle.component_paths) == set(fixture.specifications)
        assert all(status == "READY" for status in bundle.component_status.values())
        assert bundle.require("scene") == fixture.asset("scene_encoder").resolve()
        assert bundle.require("face_recognition") == fixture.asset("face_recognition").resolve()
        assert bundle.alignment_id == "arcface-5pt-112-v1"
        assert bundle.metadata == fixture.document
        assert len(bundle.bundle_id) == len(bundle.face_model_id) == 64
        int(bundle.bundle_id, 16)
        int(bundle.face_model_id, 16)
        expect_error(lambda: bundle.require("scene_decoder"), component="scene_decoder")


@test
def absent_assets_remain_declared_and_isolated():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        original = fixture.load()
        for key in fixture.specifications:
            fixture.asset(key).unlink()
        bundle = fixture.load()
        assert set(bundle.component_status.values()) == {"MODEL_MISSING"}
        assert bundle.component_paths == original.component_paths
        assert (bundle.bundle_id, bundle.face_model_id) == (original.bundle_id, original.face_model_id)
        for component in bundle.component_status:
            expect_error(lambda: bundle.require(component), "MODEL_MISSING", component)


@test
def one_missing_scene_half_does_not_disable_other_models():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        fixture.asset("scene_decoder").unlink()
        isolated(fixture.load(), "scene", "MODEL_MISSING")


@test
def changed_actual_weights_same_filename_are_not_usable():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        before = fixture.load()
        fixture.asset("face_recognition").write_bytes(b"different weights at the same filename")
        bundle = fixture.load()
        isolated(bundle, "face_recognition", "MODEL_HASH_MISMATCH")
        assert bundle.face_model_id == before.face_model_id  # Declaration is unchanged, usability is not.


@test
def invalid_graph_and_graph_declaration_mismatch_are_isolated():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        original = fixture.asset("body").read_bytes()
        fixture.asset("body").write_bytes(b"not an ONNX model")
        fixture.file("body")["sha256"] = hashlib.sha256(b"not an ONNX model").hexdigest()
        isolated(fixture.load(), "body", "MODEL_CONTRACT_INVALID")
        fixture.asset("body").write_bytes(original)
        fixture.file("body")["sha256"] = hashlib.sha256(original).hexdigest()
        fixture.file("body")["outputs"][0]["name"] = "not_the_actual_output"
        isolated(fixture.load(), "body", "MODEL_CONTRACT_INVALID")


@test
def onnx_checker_rejects_invalid_node_even_with_valid_io_and_hash():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        model = onnx.load_model(fixture.asset("body"))
        model.graph.node[0].op_type = "NotAnOnnxOperator"
        data = model.SerializeToString()
        fixture.asset("body").write_bytes(data)
        fixture.file("body")["sha256"] = hashlib.sha256(data).hexdigest()
        isolated(fixture.load(), "body", "MODEL_CONTRACT_INVALID")


@test
def manifest_cannot_lie_about_known_shapes_dtypes_or_names():
    mutations = [
        ("person", "inputs", 0, "shape", ["N", 3, 320, 320]),
        ("person", "inputs", 1, "dtype", "float32"),
        ("person", "outputs", 0, "dtype", "float32"),
        ("person", "outputs", 1, "shape", ["N", 300, 5]),
        ("person", "outputs", 2, "shape", ["N", 100]),
        ("person", "inputs", 0, "name", "bgr_yolo_images"),
        ("body", "outputs", 0, "shape", ["N", 256]),
        ("body", "inputs", 0, "dtype", "float16"),
        ("face_recognition", "inputs", 0, "shape", ["N", 3, 224, 224]),
        ("face_detection", "outputs", 0, "name", "missing_cls_8"),
        ("face_detection", "outputs", 3, "shape", [1, "cells_8", 8]),
        ("scene_encoder", "inputs", 0, "shape", [1, 3, 512, 512]),
        ("scene_decoder", "inputs", 3, "shape", [1, 1, 128, 128]),
        ("scene_decoder", "outputs", 1, "shape", [1, 1]),
    ]
    with tempfile.TemporaryDirectory() as temporary:
        for index, (key, direction, position, field, value) in enumerate(mutations):
            root = Path(temporary) / str(index)
            root.mkdir()
            fixture = BundleFixture(root)
            fixture.file(key)[direction][position][field] = value
            fixture.replace_graph(key, fixture.file(key)["inputs"], fixture.file(key)["outputs"])
            isolated(fixture.load(), "scene" if key.startswith("scene_") else key,
                     "MODEL_CONTRACT_INVALID")


@test
def batch_and_yunet_spatial_dimensions_must_be_dynamic():
    with tempfile.TemporaryDirectory() as temporary:
        for index, key in enumerate(("person", "body", "face_recognition", "face_detection")):
            root = Path(temporary) / str(index)
            root.mkdir()
            fixture = BundleFixture(root)
            file = fixture.file(key)
            file["inputs"][0]["shape"][2 if key == "face_detection" else 0] = 640 if key == "face_detection" else 1
            fixture.replace_graph(key, file["inputs"], file["outputs"])
            isolated(fixture.load(), key, "MODEL_CONTRACT_INVALID")


@test
def symbolic_dimension_names_need_not_match_manifest_text():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        fixture.file("body")["inputs"][0]["shape"][0] = "declared_batch"
        fixture.file("body")["outputs"][0]["shape"][0] = "declared_rows"
        assert fixture.load().component_status["body"] == "READY"


@test
def official_dynamic_dfine_candidates_and_decoder_mask_dimensions_are_supported():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        file = fixture.file("person")
        for output in file["outputs"]:
            output["shape"][1] = "official_topk"
        fixture.replace_graph("person", file["inputs"], file["outputs"])
        decoder = fixture.file("scene_decoder")
        for output in decoder["outputs"]:
            output["shape"][0] = "exported_batch"
            output["shape"][1] = "exported_masks"
        fixture.replace_graph("scene_decoder", decoder["inputs"], decoder["outputs"])
        assert set(fixture.load().component_status.values()) == {"READY"}
        file["exporter_revision"] = f"{'3' * 40}:{'2' * 64}"
        isolated(fixture.load(), "person", "MODEL_CONTRACT_INVALID")


@test
def global_syntax_is_validated_before_any_missing_asset():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        for key in fixture.specifications:
            fixture.asset(key).unlink()
        original = copy.deepcopy(fixture.document)
        changes = [
            lambda d: d.update(version=2),
            lambda d: d.update(version=True),
            lambda d: d.update(alignment_id="different-alignment"),
            lambda d: d.update(unexpected=True),
            lambda d: d["components"].pop("body"),
            lambda d: d["components"].update(extra={}),
            lambda d: d["components"]["body"].update(preprocessing_id="unverified"),
            lambda d: d["components"]["body"].update(files=[]),
            lambda d: d["components"]["body"]["files"][0].update(unexpected=True),
            lambda d: d["components"]["body"]["files"][0].update(inputs=[]),
            lambda d: d["components"]["body"]["files"][0]["inputs"][0].update(dtype="unregistered"),
            lambda d: d["components"]["body"]["files"][0].update(path="assets/person.onnx"),
            lambda d: d["components"]["scene"]["files"][1].update(role="encoder"),
            lambda d: d["components"]["scene"]["files"].pop(),
            lambda d: d["components"]["body"]["files"][0].update(role="encoder"),
            lambda d: d["components"]["body"]["files"][0].update(sha256="A" * 64),
            lambda d: d["components"]["body"]["files"][0].update(source_sha256="0"),
            lambda d: d["components"]["body"]["files"][0].update(source_revision="main"),
            lambda d: d["components"]["body"]["files"][0].update(exporter_revision="moving-main"),
            lambda d: d["components"]["body"]["files"][0].update(license="Apache-2.0"),
            lambda d: d["components"]["face_recognition"]["files"][0].update(exporter_revision="native:main"),
            lambda d: d["components"]["body"]["files"][0]["inputs"][0].update(shape=[None, 3, 256, 128]),
            lambda d: d["components"]["body"]["files"][0]["inputs"][0].update(shape=[0, 3, 256, 128]),
            lambda d: d["components"]["body"]["files"][0]["inputs"][0].update(shape=[True, 3, 256, 128]),
            lambda d: d["components"]["body"]["files"][0]["inputs"][0].update(shape=["", 3, 256, 128]),
            lambda d: d["components"]["body"]["files"][0]["inputs"][0].update(extra="bad"),
        ]
        for change in changes:
            fixture.document = copy.deepcopy(original)
            change(fixture.document)
            expect_error(fixture.load)


@test
def source_and_license_urls_must_be_official_https_without_host_spoofing():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        file = fixture.file("face_recognition")
        for field in ("source_url", "license_url"):
            original = file[field]
            for bad in ("http://huggingface.co/fal/AuraFace-v1/glintr100.onnx",
                        "https://huggingface.co.evil.invalid/fal/AuraFace-v1/glintr100.onnx",
                        "https://user@huggingface.co/fal/AuraFace-v1/glintr100.onnx",
                        "https://huggingface.co/unapproved/research-only/resolve/main/glintr100.onnx",
                        "https://huggingface.co/fal/AuraFace-v1/../../unapproved/weights"):
                file[field] = bad
                expect_error(fixture.load)
            file[field] = original
        file["source_url"] = file["source_url"].replace("glintr100.onnx", "scrfd_10g_bnkps.onnx")
        expect_error(fixture.load)


@test
def unsafe_declared_paths_are_global_errors_on_every_platform():
    paths = ("../outside.onnx", "/outside.onnx", "C:/outside.onnx", "C:outside.onnx",
             "\\\\host\\share\\outside.onnx", "assets/../outside.onnx", "assets\\..\\outside.onnx",
             "assets/file.onnx:stream", "assets/./body.onnx", "assets//body.onnx", "assets/zero\x00.onnx")
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        for path in paths:
            fixture.file("body")["path"] = path
            expect_error(fixture.load)


@test
def duplicate_keys_bad_json_and_missing_manifest_fail_globally():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "manifest.json"
        expect_error(lambda: ModelBundle.load(path))
        path.write_text("{", encoding="utf-8")
        expect_error(lambda: ModelBundle.load(path))
        fixture = BundleFixture(root)
        text = json.dumps(fixture.document).replace('"version": 1', '"version": 1, "version": 1', 1)
        path.write_text(text, encoding="utf-8")
        expect_error(lambda: ModelBundle.load(path))


@test
def filesystem_read_error_is_component_isolated():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        blocked = fixture.asset("body").resolve()
        original_open = Path.open

        def denied(path, *args, **kwargs):
            if path == blocked:
                raise PermissionError("isolated asset permission failure")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", denied):
            isolated(fixture.load(), "body", "MODEL_CONTRACT_INVALID")


@test
def file_and_directory_symlinks_cannot_escape_manifest_root():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for directory_link in (False, True):
            fixture_root = root / str(directory_link)
            fixture_root.mkdir()
            fixture = BundleFixture(fixture_root)
            outside = root / f"outside-{directory_link}"
            outside.mkdir()
            destination = outside / "body.onnx"
            destination.write_bytes(fixture.asset("body").read_bytes())
            link = fixture_root / "escape"
            try:
                link.symlink_to(outside if directory_link else destination,
                                target_is_directory=directory_link)
            except OSError as exc:
                raise SkipTest(f"OS disallows symlink creation: {exc}")
            fixture.file("body")["path"] = "escape/body.onnx" if directory_link else "escape"
            isolated(fixture.load(), "body", "MODEL_CONTRACT_INVALID")


@test
def unverified_external_tensor_files_are_never_loaded():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        model = onnx.load_model(fixture.asset("body"))
        external = TensorProto()
        external.name = "unverified_external"
        external.data_type = TensorProto.FLOAT
        external.dims.append(1)
        external.data_location = TensorProto.EXTERNAL
        external.external_data.add(key="location", value="../../must-not-read.bin")
        model.graph.initializer.append(external)
        data = model.SerializeToString()
        fixture.asset("body").write_bytes(data)
        fixture.file("body")["sha256"] = hashlib.sha256(data).hexdigest()
        isolated(fixture.load(), "body", "MODEL_CONTRACT_INVALID")


@test
def ids_ignore_paths_provenance_formatting_file_order_and_availability():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        before = fixture.load()
        for key in fixture.specifications:
            file = fixture.file(key)
            old_path = fixture.asset(key)
            file["path"] = "renamed/" + old_path.name
            fixture.asset(key).parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(fixture.asset(key))
            file["license_url"] += "?download=true"
            file["source_url"] += "?download=true"
            os.utime(fixture.asset(key), (1_000_000, 1_000_000))
        fixture.document["components"]["scene"]["files"].reverse()
        after = fixture.load()
        assert (after.bundle_id, after.face_model_id) == (before.bundle_id, before.face_model_id)
        fixture.asset("body").unlink()
        unavailable = fixture.load()
        assert unavailable.bundle_id == before.bundle_id
        assert unavailable.face_model_id == before.face_model_id


@test
def ids_change_for_declared_hash_and_only_faces_affect_face_id():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        for key in fixture.specifications:
            before = fixture.load()
            fixture.replace_graph(key, doc=f"distinct-artifact-{key}")
            after = fixture.load()
            assert before.bundle_id != after.bundle_id, key
            assert (before.face_model_id != after.face_model_id) == key.startswith("face_"), key


@test
def preprocessing_and_alignment_are_part_of_each_declared_vector_space_id():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        before = fixture.load()
        # A future explicitly registered preprocessing revision must change the IDs,
        # even if artifact bytes remain identical. Unknown revisions still fail syntax.
        original = subject._PREPROCESSING["face_recognition"]
        changed = "auraface-rgb112-minus127.5-div127.5-v2"
        fixture.document["components"]["face_recognition"]["preprocessing_id"] = changed
        with patch.dict(subject._PREPROCESSING, {"face_recognition": changed}):
            after = fixture.load()
        assert original != changed
        assert before.bundle_id != after.bundle_id
        assert before.face_model_id != after.face_model_id
        fixture.document["components"]["face_recognition"]["preprocessing_id"] = original
        fixture.document["alignment_id"] = "arcface-5pt-112-v2"
        with patch.object(subject, "_ALIGNMENT_ID", "arcface-5pt-112-v2"):
            realigned = fixture.load()
        assert before.bundle_id != realigned.bundle_id
        assert before.face_model_id != realigned.face_model_id


@test
def loader_does_not_create_runtime_sessions_or_use_network():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = BundleFixture(Path(temporary))
        with patch.dict(sys.modules, {"onnxruntime": None}), \
                patch("socket.create_connection", side_effect=AssertionError("network forbidden")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("download forbidden")):
            assert set(fixture.load().component_status.values()) == {"READY"}


def color_detector_fixture(fixture, *, dynamic=True):
    import numpy as np
    from onnx import numpy_helper
    inputs, outputs = fixture.specifications["person"]
    initializers = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_axis"),
        numpy_helper.from_array(np.array([300], dtype=np.int64), "count"),
        numpy_helper.from_array(np.array([4], dtype=np.int64), "four"),
        numpy_helper.from_array(np.array([[0] + [1] * 299], dtype=np.int64), "classes"),
        numpy_helper.from_array(np.tile(np.array([[[192., 192., 416., 416.]]],
                                                dtype=np.float32), (1, 300, 1)), "square"),
    ]
    if not dynamic:
        initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "one_batch"))
    nodes = [
        helper.make_node("Shape", ["images"], ["image_shape"]),
        helper.make_node("Gather", ["image_shape", "zero"], ["batch"], axis=0),
        helper.make_node("Concat", ["batch" if dynamic else "one_batch", "count"], ["rows"], axis=0),
        helper.make_node("Concat", ["rows", "four"], ["box_shape"], axis=0),
        helper.make_node("Gather", ["images", "zero"], ["red"], axis=1),
        helper.make_node("ReduceMean", ["red"], ["red_mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Unsqueeze", ["red_mean", "one_axis"], ["score_column"]),
        helper.make_node("Identity", ["score_column"], ["score_batch"]) if dynamic else
        helper.make_node("Gather", ["score_column", "zero"], ["score_batch"], axis=0),
        helper.make_node("Expand", ["score_batch", "rows"], ["scores"]),
        helper.make_node("Expand", ["classes", "rows"], ["labels"]),
        helper.make_node("Expand", ["square", "box_shape"], ["boxes"]),
    ]
    info = lambda value: helper.make_tensor_value_info(
        value["name"], TensorProto.FLOAT if value["dtype"] == "float32" else TensorProto.INT64,
        value["shape"])
    graph = helper.make_graph(nodes, "color-and-coordinate-consumer",
                              [info(value) for value in inputs], [info(value) for value in outputs],
                              initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=10)
    data = model.SerializeToString()
    fixture.asset("person").write_bytes(data)
    fixture.file("person")["sha256"] = hashlib.sha256(data).hexdigest()
    sys.path.insert(0, str(SOURCE.parents[2]))
    from app.ai.detector import PersonDetector
    return PersonDetector(fixture.load(), ["CPUExecutionProvider"], confidence=.1)


@test
def dfine_batch_preserves_rgb_native_boxes_and_distinct_input_order():
    import numpy as np
    with tempfile.TemporaryDirectory() as directory:
        detector = color_detector_fixture(BundleFixture(Path(directory)))
        landscape = np.zeros((100, 200, 3), dtype=np.uint8)
        landscape[:, :, 2] = 255
        portrait = np.zeros((200, 100, 3), dtype=np.uint8)
        portrait[:, :, 2] = 255
        blue = np.zeros((100, 200, 3), dtype=np.uint8)
        blue[:, :, 0] = 255
        landscape.setflags(write=False)
        portrait.setflags(write=False)
        assert detector.detect_batch([]) == []
        single = detector.detect(landscape)
        assert len(single) == 1
        np.testing.assert_allclose(single[0][0], [60, 10, 130, 80], atol=1e-4)
        outcomes = detector.detect_batch([landscape, portrait, blue, landscape])
        assert [len(rows) for rows in outcomes] == [1, 1, 0, 1]
        np.testing.assert_allclose(outcomes[0][0][0], [60, 10, 130, 80], atol=1e-4)
        np.testing.assert_allclose(outcomes[1][0][0], [10, 60, 80, 130], atol=1e-4)
        np.testing.assert_allclose(outcomes[3][0][0], [60, 10, 130, 80], atol=1e-4)
        assert .49 < outcomes[0][0][1] < .51


@test
def dfine_rejects_actual_batch_shape_mismatch_instead_of_dropping_a_camera():
    import numpy as np
    with tempfile.TemporaryDirectory() as directory:
        detector = color_detector_fixture(BundleFixture(Path(directory)), dynamic=False)
        from app.ai.model_bundle import ModelBundleError as DetectorError
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        try:
            detector.detect_batch([frame, frame])
        except DetectorError as exc:
            assert exc.code == "MODEL_CONTRACT_INVALID" and exc.component == "person"
        else:
            raise AssertionError("A malformed batch silently dropped a camera")


if __name__ == "__main__":
    failed = skipped = 0
    for case in TESTS:
        try:
            case()
            print("PASS", case.__name__)
        except SkipTest as exc:
            skipped += 1
            print("SKIP", case.__name__, exc)
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"{len(TESTS)-failed-skipped}/{len(TESTS)} passed; {skipped} skipped")
    raise SystemExit(bool(failed))
