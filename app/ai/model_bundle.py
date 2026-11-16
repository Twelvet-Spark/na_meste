"""Validate a declared, locally prepared ONNX bundle without creating sessions.

Manifest/provenance errors invalidate the manifest. Asset failures invalidate only
that component; callers must use ``require`` before constructing a model session.
IDs describe declared vector spaces, never file locations or file availability.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from urllib.parse import unquote, urlsplit

import onnx
from onnx import TensorProto

_ALIGNMENT_ID = "arcface-5pt-112-v1"
_PREPROCESSING = {
    "person": "dfine-rgb-letterbox640-black-div255-v1",
    "face_detection": "yunet-bgr-f32-strides8-16-32-v1",
    "face_recognition": "auraface-rgb112-minus127.5-div127.5-v1",
    "body": "osnet-rgb256x128-imagenet-v1",
    "scene": "mobilesam-rgb-longest1024-meanstd-pad-v1",
}
_LICENSES = {"person": "Apache-2.0", "face_detection": "MIT",
             "face_recognition": "Apache-2.0", "body": "MIT", "scene": "Apache-2.0"}
_FILE_FIELDS = {"role", "path", "sha256", "source_url", "source_revision",
                "source_sha256", "license", "license_url", "exporter_revision", "inputs", "outputs"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_EXPORTER = re.compile(r"[0-9a-f]{40}:[0-9a-f]{64}\Z")
_DFINE_EXPORTER = "956d1709314c2c6a4df6f34de232054578a7449f"
_SCENE_EXPORTER = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
_DFINE_RELEASE = "release:177720969:asset:196374453"
_BODY_FILENAME = ("osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_"
                  "b64_fb10_softmax_labsmth_flip_jitter.pth")
_DTYPES = {
    "float32": TensorProto.FLOAT, "int64": TensorProto.INT64,
    "float16": TensorProto.FLOAT16, "float64": TensorProto.DOUBLE,
    "int8": TensorProto.INT8, "uint8": TensorProto.UINT8,
    "int16": TensorProto.INT16, "uint16": TensorProto.UINT16,
    "int32": TensorProto.INT32, "uint32": TensorProto.UINT32,
    "uint64": TensorProto.UINT64, "bool": TensorProto.BOOL,
    "bfloat16": TensorProto.BFLOAT16,
}
_DTYPE_NAMES = {value: key for key, value in _DTYPES.items()}


class ModelBundleError(RuntimeError):
    """Static manifest or component failure, safe to inspect before session creation."""

    def __init__(self, code: str, component: str | None, detail: str = ""):
        self.code = code
        self.component = component
        super().__init__(f"{code} [{component or 'manifest'}]" + (f": {detail}" if detail else ""))


def _invalid(detail: str, component: str | None = None) -> ModelBundleError:
    return ModelBundleError("MODEL_CONTRACT_INVALID", component, detail)


def _exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise _invalid(f"Invalid {label} fields")


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip()) and not any(ord(c) < 32 for c in value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _invalid("Duplicate JSON key")
        result[key] = value
    return result


def _relative_path(value):
    # Enforce one portable spelling, including Windows drives, UNC and ADS, even
    # when a bundle is inspected on a different OS from its eventual deployment.
    if not _nonempty(value) or "\\" in value or ":" in value:
        raise _invalid("Asset path must be a portable relative path")
    parts = value.split("/")
    if (PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in ("", ".", "..") for part in parts)
            or any(part.rstrip(" .") != part for part in parts)):
        raise _invalid("Asset path is not contained")


def _official_url(value, component, *, source, revision):
    if not _nonempty(value):
        raise _invalid("Missing provenance URL")
    try:
        url = urlsplit(value)
        if (url.scheme != "https" or not url.hostname or url.username is not None
                or url.password is not None or url.port is not None or url.fragment):
            raise ValueError("Invalid HTTPS origin")
        path = unquote(url.path)
        if "\\" in path or any(part in (".", "..") for part in path.split("/")):
            raise ValueError("Invalid URL path")
    except ValueError as exc:
        raise _invalid("Invalid provenance URL") from exc

    if source:
        expected = {
            "person": ("github.com", "/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth"),
            "face_detection": ("media.githubusercontent.com",
                               f"/media/opencv/opencv_zoo/{revision}/models/face_detection_yunet/face_detection_yunet_2026may.onnx"),
            "face_recognition": ("huggingface.co", f"/fal/AuraFace-v1/resolve/{revision}/glintr100.onnx"),
            "body": ("huggingface.co", f"/kaiyangzhou/osnet/resolve/{revision}/{_BODY_FILENAME}"),
            "scene": ("raw.githubusercontent.com", f"/ChaoningZhang/MobileSAM/{revision}/weights/mobile_sam.pt"),
        }[component]
        if (url.hostname, path) != expected:
            raise _invalid("Source is not the selected official artifact")
        return

    repositories = {
        "person": ("Peterande/D-FINE",),
        "face_detection": ("opencv/opencv_zoo",),
        "face_recognition": ("fal/AuraFace-v1",),
        "body": ("kaiyangzhou/osnet", "KaiyangZhou/deep-person-reid"),
        "scene": ("ChaoningZhang/MobileSAM",),
    }[component]
    if url.hostname not in ("github.com", "raw.githubusercontent.com", "huggingface.co"):
        raise _invalid("License URL is not official upstream")
    if not any(path.startswith(f"/{repository}/") for repository in repositories):
        raise _invalid("License URL is not the selected upstream repository")
    # Do not accept a repository path on a different service as an official source.
    huggingface_repositories = {"fal/AuraFace-v1", "kaiyangzhou/osnet"}
    allowed = [repository for repository in repositories
               if (repository in huggingface_repositories) == (url.hostname == "huggingface.co")]
    if not any(path.startswith(f"/{repository}/") for repository in allowed):
        raise _invalid("License URL has the wrong upstream host")


def _validate_io_syntax(items):
    if not isinstance(items, list) or not items:
        raise _invalid("IO declarations must be nonempty arrays")
    names = set()
    for item in items:
        _exact_keys(item, {"name", "dtype", "shape"}, "tensor")
        if not _nonempty(item["name"]) or item["name"] in names:
            raise _invalid("Invalid or duplicate tensor name")
        names.add(item["name"])
        if not isinstance(item["dtype"], str) or item["dtype"] not in _DTYPES:
            raise _invalid("Unknown tensor dtype")
        if not isinstance(item["shape"], list) or not item["shape"]:
            raise _invalid("Missing tensor dimensions")
        for dimension in item["shape"]:
            if not ((type(dimension) is int and dimension > 0) or _nonempty(dimension)):
                raise _invalid("Dimensions must be positive integers or symbolic strings")


def _validate_manifest(document):
    _exact_keys(document, {"version", "alignment_id", "components"}, "manifest")
    if type(document["version"]) is not int or document["version"] != 1:
        raise _invalid("Unsupported manifest version")
    if document["alignment_id"] != _ALIGNMENT_ID:
        raise _invalid("Unsupported face alignment")
    _exact_keys(document["components"], _PREPROCESSING, "components")
    paths = set()
    for component, declaration in document["components"].items():
        _exact_keys(declaration, {"preprocessing_id", "files"}, "component")
        if declaration["preprocessing_id"] != _PREPROCESSING[component]:
            raise _invalid("Unsupported preprocessing")
        files = declaration["files"]
        roles = {"encoder", "decoder"} if component == "scene" else {"model"}
        if not isinstance(files, list) or len(files) != len(roles):
            raise _invalid("Every component artifact must be declared")
        found = set()
        for file in files:
            _exact_keys(file, _FILE_FIELDS, "artifact")
            if not isinstance(file["role"], str) or file["role"] not in roles or file["role"] in found:
                raise _invalid("Invalid or duplicate artifact role")
            found.add(file["role"])
            _relative_path(file["path"])
            portable_path = file["path"].casefold()
            if portable_path in paths:
                raise _invalid("Duplicate artifact path")
            paths.add(portable_path)
            for key in ("sha256", "source_sha256"):
                if not isinstance(file[key], str) or not _HASH.fullmatch(file[key]):
                    raise _invalid("Invalid SHA256 declaration")
            revision = file["source_revision"]
            if not isinstance(revision, str) or not (
                    _REVISION.fullmatch(revision) or (component == "person" and revision == _DFINE_RELEASE)):
                raise _invalid("Source revision must identify an immutable artifact")
            if file["license"] != _LICENSES[component]:
                raise _invalid("License does not match the selected component")
            _official_url(file["source_url"], component, source=True, revision=revision)
            _official_url(file["license_url"], component, source=False, revision=revision)
            exporter = file["exporter_revision"]
            if not isinstance(exporter, str):
                raise _invalid("Missing exporter identity")
            if (component == "face_detection"
                    or (component == "face_recognition" and exporter == f"native:{revision}")):
                if exporter != f"native:{revision}":
                    raise _invalid("Native ONNX provenance must identify its source revision")
                if file["source_sha256"] != file["sha256"]:
                    raise _invalid("Native ONNX artifact must retain its source hash")
            elif not _EXPORTER.fullmatch(exporter):
                raise _invalid("Converted artifact must identify upstream and local exporter hashes")
            elif component == "face_recognition" and not exporter.startswith(f"{revision}:"):
                raise _invalid("Prepared AuraFace must identify its recognition source revision")
            _validate_io_syntax(file["inputs"])
            _validate_io_syntax(file["outputs"])


def _identifier(document, components):
    identity = {
        "alignment_id": document["alignment_id"],
        "components": {
            component: {
                "preprocessing_id": document["components"][component]["preprocessing_id"],
                "files": {file["role"]: file["sha256"]
                          for file in document["components"][component]["files"]},
            }
            for component in components
        },
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _graph_io(values):
    result = []
    for value in values:
        if not value.type.HasField("tensor_type"):
            raise ValueError("Only tensor IO is supported")
        tensor = value.type.tensor_type
        if tensor.elem_type not in _DTYPE_NAMES or not tensor.HasField("shape"):
            raise ValueError("Missing tensor dtype or shape")
        shape = []
        for dimension in tensor.shape.dim:
            if dimension.HasField("dim_value") and dimension.dim_value > 0:
                shape.append(dimension.dim_value)
            elif dimension.HasField("dim_param") and _nonempty(dimension.dim_param):
                shape.append(dimension.dim_param)
            else:
                raise ValueError("Missing or invalid graph dimension")
        result.append({"name": value.name, "dtype": _DTYPE_NAMES[tensor.elem_type], "shape": shape})
    return result


def _same_shape(actual, declared):
    return len(actual) == len(declared) and all(
        (isinstance(a, str) and isinstance(d, str)) or a == d
        for a, d in zip(actual, declared))


def _match_declarations(actual, declared):
    if len(actual) != len(declared):
        raise ValueError("Graph and declared IO counts differ")
    actual_by_name = {item["name"]: item for item in actual}
    if len(actual_by_name) != len(actual):
        raise ValueError("Duplicate graph IO")
    for item in declared:
        graph = actual_by_name.get(item["name"])
        if (graph is None or graph["dtype"] != item["dtype"]
                or not _same_shape(graph["shape"], item["shape"])):
            raise ValueError("Graph IO does not match the manifest")


def _tensor_contract(item, shape, *, dtype="float32", symbolic_fixed=()):
    if item["dtype"] != dtype or len(item["shape"]) != len(shape):
        raise ValueError("Unsupported model tensor dtype or rank")
    for index, (actual, expected) in enumerate(zip(item["shape"], shape)):
        if expected == "dynamic":
            valid = isinstance(actual, str)
        elif expected == "positive":
            valid = isinstance(actual, str) or (type(actual) is int and actual > 0)
        else:
            valid = actual == expected or (index in symbolic_fixed and isinstance(actual, str))
        if not valid:
            raise ValueError("Unsupported model tensor dimension")


def _known_contract(component, file, inputs, outputs):
    by_input = {item["name"]: item for item in inputs}
    by_output = {item["name"]: item for item in outputs}
    if component == "person":
        if set(by_input) != {"images", "orig_target_sizes"} or set(by_output) != {"labels", "boxes", "scores"}:
            raise ValueError("D-FINE names do not match the official export")
        _tensor_contract(by_input["images"], ["dynamic", 3, 640, 640])
        _tensor_contract(by_input["orig_target_sizes"], ["dynamic", 2], dtype="int64")
        official = file["exporter_revision"].split(":", 1)[0] == _DFINE_EXPORTER
        for name, shape, dtype in (("labels", ["dynamic", 300], "int64"),
                                   ("boxes", ["dynamic", 300, 4], "float32"),
                                   ("scores", ["dynamic", 300], "float32")):
            _tensor_contract(by_output[name], shape, dtype=dtype, symbolic_fixed=(1,) if official else ())
    elif component == "face_detection":
        if len(inputs) != 1:
            raise ValueError("YuNet requires one image input")
        _tensor_contract(inputs[0], [1, 3, "dynamic", "dynamic"])
        expected = {f"{head}_{stride}" for stride in (8, 16, 32) for head in ("cls", "obj", "bbox", "kps")}
        if set(by_output) != expected:
            raise ValueError("YuNet requires every named stride head")
        for stride in (8, 16, 32):
            for head, channels in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)):
                _tensor_contract(by_output[f"{head}_{stride}"], [1, "positive", channels])
    elif component in ("face_recognition", "body"):
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Embedding model requires exactly one input and output")
        height, width = (112, 112) if component == "face_recognition" else (256, 128)
        _tensor_contract(inputs[0], ["dynamic", 3, height, width])
        _tensor_contract(outputs[0], ["dynamic", 512])
    elif file["role"] == "encoder":
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("MobileSAM encoder requires exactly one input and output")
        _tensor_contract(inputs[0], [1, 3, 1024, 1024])
        _tensor_contract(outputs[0], [1, 256, 64, 64])
    else:
        expected_inputs = {
            "image_embeddings": [1, 256, 64, 64],
            "point_coords": [1, "dynamic", 2], "point_labels": [1, "dynamic"],
            "mask_input": [1, 1, 256, 256], "has_mask_input": [1], "orig_im_size": [2],
        }
        expected_outputs = {"masks": [1, 4, "positive", "positive"],
                            "iou_predictions": [1, 4], "low_res_masks": [1, 4, 256, 256]}
        if set(by_input) != set(expected_inputs) or set(by_output) != set(expected_outputs):
            raise ValueError("MobileSAM decoder names do not match the official export")
        for name, shape in expected_inputs.items():
            _tensor_contract(by_input[name], shape)
        official = file["exporter_revision"].split(":", 1)[0] == _SCENE_EXPORTER
        for name, shape in expected_outputs.items():
            # The official tracer may leave batch/mask/spatial output axes symbolic.
            _tensor_contract(by_output[name], shape,
                             symbolic_fixed=tuple(range(len(shape))) if official else ())


def _reject_external_data(message):
    # Walk graph attributes/functions too, not only top-level initializers. The
    # manifest has no independently hashed external tensor files in version 1.
    if isinstance(message, TensorProto) and (
            message.data_location == TensorProto.EXTERNAL or message.external_data):
        raise ValueError("External tensor data is not covered by the manifest")
    for descriptor, value in message.ListFields():
        if descriptor.message_type is not None:
            if descriptor.is_repeated:
                for child in value:
                    _reject_external_data(child)
            else:
                _reject_external_data(value)


def _validate_asset(root, file, component):
    path = root / file["path"]
    try:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise _invalid("Asset symlink escapes the bundle", component)
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise _invalid("Asset is not a regular file", component)
        with resolved.open("rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise _invalid("Asset is not a regular file", component)
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != file["sha256"]:
                raise ModelBundleError("MODEL_HASH_MISMATCH", component, "Declared artifact hash differs")
            stream.seek(0)
            model = onnx.load_model(stream, load_external_data=False)
        # Validate exactly the bytes hashed above; never ask ONNX to resolve more
        # files or create runtime sessions as part of manifest loading.
        _reject_external_data(model)
        onnx.checker.check_model(model)
        inputs, outputs = _graph_io(model.graph.input), _graph_io(model.graph.output)
        _match_declarations(inputs, file["inputs"])
        _match_declarations(outputs, file["outputs"])
        _known_contract(component, file, inputs, outputs)
        return resolved
    except ModelBundleError:
        raise
    except FileNotFoundError as exc:
        raise ModelBundleError("MODEL_MISSING", component, "Declared artifact is absent") from exc
    except Exception as exc:
        raise _invalid("Artifact cannot be read or violates its ONNX contract", component) from exc


class ModelBundle:
    """Declared provenance and independently validated component readiness.

    ``component_paths`` has person/face_detection/face_recognition/body plus
    scene_encoder/scene_decoder. ``require('scene')`` returns the encoder only
    after both scene artifacts validate. No other path aliases are provided.
    """

    def __init__(self, document, paths, errors):
        self.alignment_id = document["alignment_id"]
        self.metadata = document
        self.bundle_id = _identifier(document, _PREPROCESSING)
        self.face_model_id = _identifier(document, ("face_detection", "face_recognition"))
        self.component_paths: dict[str, Path] = paths
        self._errors: dict[str, ModelBundleError] = errors
        self.component_status: dict[str, str] = {
            component: errors[component].code if component in errors else "READY"
            for component in _PREPROCESSING
        }

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        try:
            manifest = Path(path).resolve()
            with manifest.open("r", encoding="utf-8") as stream:
                document = json.load(stream, object_pairs_hook=_unique_object)
            _validate_manifest(document)
        except ModelBundleError:
            raise
        except Exception as exc:
            raise _invalid("Manifest cannot be read or parsed") from exc

        root = manifest.parent
        paths, errors = {}, {}
        for component in _PREPROCESSING:
            files = sorted(document["components"][component]["files"], key=lambda item: item["role"])
            for file in files:
                key = f"scene_{file['role']}" if component == "scene" else component
                paths[key] = root / file["path"]
                try:
                    paths[key] = _validate_asset(root, file, component)
                except ModelBundleError as exc:
                    # A deterministic first error is retained when both scene
                    # halves fail; no component can become READY after a failure.
                    errors.setdefault(component, exc)
        return cls(document, paths, errors)

    def require(self, component: str) -> Path:
        if component not in self.component_status:
            raise _invalid("Unknown model component", component)
        if component in self._errors:
            raise self._errors[component]
        return self.component_paths["scene_encoder" if component == "scene" else component]
