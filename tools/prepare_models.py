"""Prepare the approved commercial-use ONNX bundle, never during server startup.

Use the isolated .model-build environment; this module's import is stdlib-only.
All upstream downloads are immutable/hash-locked. A manifest is published only
following checker, actual inference, and source/runtime parity validation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import time
import types
import urllib.parse
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = Path(__file__).with_name("model_build_lock.json")
OFFICIAL_HOSTS = {"github.com", "api.github.com", "raw.githubusercontent.com",
                  "media.githubusercontent.com", "codeload.github.com", "huggingface.co"}
ALIGNMENT_ID = "arcface-5pt-112-v1"
_DLL_HANDLES = []


class BuildError(RuntimeError):
    """Preparation failed; the last published manifest remains usable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, spec: dict) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    if "size" in spec and path.stat().st_size != spec["size"]:
        return False
    if sha256_file(path) != spec["sha256"]:
        return False
    if "git_blob_sha1" in spec:
        digest = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != spec["git_blob_sha1"]:
            return False
    return True


def download_verified(spec: dict, cache: Path) -> Path:
    """Reuse only authenticated cache bytes; download to an unpublishable temp."""
    parsed = urllib.parse.urlsplit(spec["url"])
    if (parsed.scheme != "https" or parsed.hostname not in OFFICIAL_HOSTS
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])):
        raise BuildError("Unapproved source URL or SHA256")
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / spec["sha256"]
    if _verified(destination, spec):
        return destination
    # Also reuse the human-readable filenames from initial authentication.
    named = cache / spec.get("cache_name", spec["sha256"])
    if named != destination and _verified(named, spec):
        return named
    descriptor, name = tempfile.mkstemp(prefix="download-", suffix=".part", dir=cache)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "na-meste-model-builder/1"})
            with urllib.request.urlopen(request, timeout=180) as response:
                # Official GitHub/HF HTTPS CDN redirects are expected; never downgrade.
                if hasattr(response, "geturl") and urllib.parse.urlsplit(response.geturl()).scheme != "https":
                    raise BuildError("Insecure artifact redirect")
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > spec.get("size", 1024 * 1024 * 1024):
                        raise BuildError("Artifact exceeds declared size")
                    stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if not _verified(temporary, spec):
            raise BuildError(f"Artifact hash/size/blob mismatch: {spec['url']}")
        os.replace(temporary, destination)
        return destination
    except (OSError, ValueError) as exc:
        raise BuildError(f"Artifact download failed: {spec['url']}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _relative_file(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ":" in name
            or any(part in (".", "..") for part in name.split("/"))):
        raise BuildError(f"Unsafe artifact path: {name}")
    candidate = root.joinpath(*path.parts)
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise BuildError(f"Artifact path escapes bundle: {name}")
    current = candidate
    while current != root:
        if current.is_symlink():
            raise BuildError(f"Symlink in artifact path: {name}")
        current = current.parent
    return candidate


def extract_source(archive: Path, destination: Path) -> Path:
    """Extract authenticated source afresh; cached extracted code is never trusted."""
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if len(members) > 10000 or sum(item.file_size for item in members) > 512 * 1024 * 1024:
            raise BuildError("Source archive exceeds bounds")
        roots = set()
        for member in members:
            name = member.filename.rstrip("/")
            target = _relative_file(destination, name)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise BuildError("Source archive contains symlink")
            roots.add(PurePosixPath(name).parts[0])
            if not member.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        if len(roots) != 1:
            raise BuildError("Expected one source archive root")
    return destination / roots.pop()


def publish_bundle(stage: Path, output: Path, manifest: dict, generation: str) -> Path:
    """Atomic manifest pointer cutover; never rename away a working old bundle.

    Immutable generations also keep sessions/readers using the prior manifest
    valid. Old generations are deliberately not deleted by the builder.
    """
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", generation):
        raise BuildError("Invalid generation")
    published = deepcopy(manifest)
    for component in published["components"].values():
        for item in component["files"]:
            path = _relative_file(stage, item["path"])
            if not _verified(path, {"sha256": item["sha256"]}):
                raise BuildError(f"Unverified artifact at publication: {item['path']}")
            item["path"] = f"generations/{generation}/{item['path']}"
    if output.is_symlink():
        raise BuildError("Output directory cannot be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    generations = output / "generations"
    if generations.is_symlink():
        raise BuildError("Generations directory cannot be a symlink")
    generations.mkdir(exist_ok=True)
    target = generations / generation
    if target.exists():
        raise BuildError("Generation already exists")
    temporary = output / f".manifest-{generation}.tmp"
    moved = False
    try:
        os.replace(stage, target)
        moved = True
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(published, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / "manifest.json")
    except BaseException:
        if moved:
            shutil.rmtree(target)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return target / "build-report.json"


@contextmanager
def _source_packages(packages: dict[str, Path]):
    """Register inference-only upstream packages without training/UI __init__.

    Actual architecture/core modules remain byte-for-byte official sources.
    This omits D-FINE's dataset/loss/optimizer registration and MobileSAM's
    automatic-mask-generator/UI imports; none participates in these exports.
    """
    prefixes = tuple(packages)
    saved = {key: value for key, value in sys.modules.items()
             if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)}
    for key in saved:
        del sys.modules[key]
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    try:
        yield
    finally:
        for key in list(sys.modules):
            if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes):
                del sys.modules[key]
        sys.modules.update(saved)


@contextmanager
def _no_pretrained_downloads(torch):
    from unittest.mock import patch
    def forbidden(*args, **kwargs):
        raise BuildError("Unexpected pretrained download; only locked artifacts are allowed")
    with patch.object(torch.hub, "load_state_dict_from_url", forbidden), \
            patch.object(torch.hub, "download_url_to_file", forbidden), \
            patch.object(urllib.request, "urlopen", forbidden):
        yield


def _state_dict(torch, value, *, buffers=None):
    if not isinstance(value, dict) or not value:
        raise BuildError("Checkpoint does not contain a nonempty state dictionary")
    buffers = {} if buffers is None else buffers
    state = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise BuildError("Checkpoint has invalid state")
        name = name.removeprefix("module.")
        if name in state:
            raise BuildError("Duplicate checkpoint tensor name")
        expected = buffers.get(name)
        if name in buffers:
            # Architecture buffers constrain structure, not their finite
            # checkpoint values. Preserve those values exactly; only the
            # nonfinite sentinel positions/signs must match generated state.
            if (not isinstance(expected, torch.Tensor)
                    or tensor.dtype != expected.dtype or tensor.shape != expected.shape
                    or torch.isnan(tensor).any() or torch.isnan(expected).any()
                    or not torch.equal(torch.isposinf(tensor), torch.isposinf(expected))
                    or not torch.equal(torch.isneginf(tensor), torch.isneginf(expected))):
                raise BuildError(f"Checkpoint has invalid structural buffer: {name}")
        elif not torch.isfinite(tensor).all():
            raise BuildError(f"Checkpoint has invalid/nonfinite parameter or undeclared buffer: {name}")
        state[name] = tensor
    return state


def _export(torch, module, args, destination, inputs, outputs, dynamic_axes):
    import onnx
    module.eval()
    with torch.inference_mode():
        examples = module(*args)
        if isinstance(examples, torch.Tensor):
            examples = (examples,)
        torch.onnx.export(module, args, str(destination), input_names=inputs,
                          output_names=outputs, dynamic_axes=dynamic_axes,
                          opset_version=16, dynamo=False, export_params=True,
                          do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL)
    # The legacy exporter sometimes leaves fixed TopK/reshape dimensions
    # symbolic. Record observed dimensions and preserve only intended dynamic
    # axes; N=1/N=4 and varying prompt counts are independently checked in ORT.
    graph = onnx.load(str(destination))
    for declaration, example in zip(graph.graph.output, examples, strict=True):
        dimensions = declaration.type.tensor_type.shape.dim
        for axis, size in enumerate(example.shape):
            dimension = dimensions[axis]
            dimension.Clear()
            symbol = dynamic_axes.get(declaration.name, {}).get(axis)
            if symbol:
                dimension.dim_param = symbol
            else:
                dimension.dim_value = int(size)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(destination))


def _export_dfine(torch, model, destination):
    # Trace the official decoder's batch>1 anchor-repeat branch. Its repeat
    # factor remains derived from runtime N; tracing N=1 omits it entirely.
    # Four bounds export memory while exercising the same branch as upstream's
    # N=32 example. Actual N=1/N=4 source/ORT parity still gates publication.
    args = (torch.zeros(4, 3, 640, 640), torch.full((4, 2), 640, dtype=torch.int64))
    axes = {name: {0: "N"} for name in ("images", "orig_target_sizes", "labels", "boxes", "scores")}
    _export(torch, model, args, destination, ["images", "orig_target_sizes"], ["labels", "boxes", "scores"], axes)


def _dfine(torch, root, checkpoint, destination):
    packages = {"src": root / "src", "src.nn": root / "src/nn",
                "src.nn.backbone": root / "src/nn/backbone", "src.zoo": root / "src/zoo",
                "src.zoo.dfine": root / "src/zoo/dfine"}
    with _source_packages(packages), _no_pretrained_downloads(torch):
        core = importlib.import_module("src.core")
        for name in ("src.nn.backbone.hgnetv2", "src.zoo.dfine.dfine",
                     "src.zoo.dfine.dfine_decoder", "src.zoo.dfine.hybrid_encoder",
                     "src.zoo.dfine.postprocessor"):
            importlib.import_module(name)
        cfg = core.YAMLConfig(str(root / "configs/dfine/dfine_hgnetv2_s_coco.yml"))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint_data, dict):
            raise BuildError("Invalid D-FINE checkpoint")
        if "ema" in checkpoint_data:
            state = checkpoint_data["ema"]["module"]
        else:
            state = checkpoint_data["model"]
        source_model = cfg.model
        source_model.load_state_dict(
            _state_dict(torch, state, buffers=dict(source_model.named_buffers())), strict=True)
        source_model.eval()
        class OfficialDeploy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = source_model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
            def forward(self, images, orig_target_sizes):
                return self.postprocessor(self.model(images), orig_target_sizes)
        model = OfficialDeploy().cpu().eval()
        _export_dfine(torch, model, destination)
    return model


def _osnet(torch, root, checkpoint, destination):
    spec = importlib.util.spec_from_file_location("_approved_osnet_ain", root / "torchreid/models/osnet_ain.py")
    architecture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(architecture)
    with _no_pretrained_downloads(torch):
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = _state_dict(torch, raw.get("state_dict", raw) if isinstance(raw, dict) else raw)
        classifier = state.get("classifier.weight")
        if classifier is None or classifier.ndim != 2 or classifier.shape[1] != 512:
            raise BuildError("OSNet checkpoint classifier does not match author architecture")
        model = architecture.osnet_ain_x1_0(num_classes=classifier.shape[0], pretrained=False).cpu().eval()
        model.load_state_dict(state, strict=True)
        class NormalizedEmbedding(torch.nn.Module):
            def __init__(self, body):
                super().__init__()
                self.body = body
            def forward(self, images):
                return torch.nn.functional.normalize(self.body(images), p=2, dim=1)
        model = NormalizedEmbedding(model).eval()
        _export(torch, model, (torch.zeros(1, 3, 256, 128),), destination,
                ["images"], ["embeddings"], {"images": {0: "N"}, "embeddings": {0: "N"}})
    return model


def _mobilesam(torch, root, checkpoint, encoder_path, decoder_path):
    with _source_packages({"mobile_sam": root / "mobile_sam"}), _no_pretrained_downloads(torch):
        registry = importlib.import_module("mobile_sam.build_sam").sam_model_registry
        wrapper = importlib.import_module("mobile_sam.utils.onnx").SamOnnxModel
        # Explicit safe state loading, rather than the upstream torch.load default.
        model = registry["vit_t"](checkpoint=None).cpu().eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(_state_dict(torch, state), strict=True)
        encoder = model.image_encoder.cpu().eval()
        decoder = wrapper(model, return_single_mask=False).cpu().eval()
        _export(torch, encoder, (torch.zeros(1, 3, 1024, 1024),), encoder_path,
                ["images"], ["image_embeddings"], {})
        args = (torch.zeros(1, 256, 64, 64), torch.tensor([[[512., 512.], [0., 0.]]]),
                torch.tensor([[1., -1.]]), torch.zeros(1, 1, 256, 256),
                torch.zeros(1), torch.tensor([360., 640.]))
        names = ["image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input", "orig_im_size"]
        axes = {"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"},
                "masks": {2: "mask_height", 3: "mask_width"}}
        _export(torch, decoder, args, decoder_path, names,
                ["masks", "iou_predictions", "low_res_masks"], axes)
    return encoder, decoder


def _graph_io(path, component, role="model"):
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise BuildError("External ONNX tensor data is not supported by bundle v1")
    onnx.checker.check_model(model, full_check=True)
    def declarations(items):
        result = []
        for item in items:
            tensor = item.type.tensor_type
            dtype = onnx.TensorProto.DataType.Name(tensor.elem_type).lower()
            dtype = {"float": "float32", "double": "float64", "float16": "float16"}.get(dtype, dtype)
            shape = []
            for dim in tensor.shape.dim:
                if dim.HasField("dim_value") and dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    raise BuildError(f"Undeclared ONNX dimension: {path.name}/{item.name}")
            result.append({"name": item.name, "dtype": dtype, "shape": shape})
        return result
    inputs, outputs = declarations(model.graph.input), declarations(model.graph.output)
    _validate_io(component, role, inputs, outputs)
    return inputs, outputs


def _scene_vocabulary():
    # Loading pure data by path must not initialize the application/configuration.
    spec = importlib.util.spec_from_file_location("scene_vocabulary", ROOT / "app/ai/scene_vocabulary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _owlv2(torch, auxiliary_dir, weights, destination):
    from unittest.mock import patch
    import onnx
    vocabulary = _scene_vocabulary()
    with tempfile.TemporaryDirectory(prefix="owlv2-") as directory, \
            patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
        directory = Path(directory)
        for source in auxiliary_dir.iterdir():
            shutil.copyfile(source, directory / source.name)
        shutil.copyfile(weights, directory / "model.safetensors")
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        model = Owlv2ForObjectDetection.from_pretrained(
            directory, local_files_only=True, attn_implementation="eager").float().eval()
        processor = Owlv2Processor.from_pretrained(directory, local_files_only=True)
        tokens = processor(text=[[prompt for prompt, _, _ in vocabulary.VOCABULARY]],
                           return_tensors="pt")
        with torch.inference_mode():
            # transformers 5.23.0 Owlv2Model.forward: projected pooled text,
            # L2-normalized before image_text_embedder returns query_embeds.
            text = model.owlv2.text_model(input_ids=tokens["input_ids"],
                                         attention_mask=tokens.get("attention_mask")).pooler_output
            queries = model.owlv2.text_projection(text)
            queries = queries / torch.linalg.vector_norm(queries, ord=2, dim=-1, keepdim=True)
            queries = queries.reshape(1, len(vocabulary.VOCABULARY), -1)
        class FixedVocabulary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.register_buffer("query_embeds", queries)
                self.register_buffer("query_mask", torch.ones(1, len(vocabulary.VOCABULARY), dtype=torch.bool))

            def forward(self, pixel_values):
                feature_map = self.model.image_embedder(pixel_values)[0]
                image_feats = feature_map.reshape(1, -1, feature_map.shape[-1])
                logits, _ = self.model.class_predictor(image_feats, self.query_embeds, self.query_mask)
                boxes = self.model.box_predictor(image_feats, feature_map)
                return logits, boxes
        wrapper = FixedVocabulary().eval()
        _export(torch, wrapper, (torch.zeros(1, 3, 960, 960),), destination,
                ["pixel_values"], ["logits", "boxes"], {})
    graph = onnx.load(str(destination))
    metadata = graph.metadata_props.add()
    metadata.key = "na_meste.scene_vocabulary"
    metadata.value = vocabulary.VOCABULARY_ID
    onnx.save(graph, str(destination))
    return wrapper


def _validate_io(component, role, inputs, outputs):
    def check(item, dtype, shape):
        actual = item["shape"]
        if item["dtype"] != dtype or len(actual) != len(shape):
            raise BuildError(f"Wrong {component}/{role} IO type/rank")
        for value, expected in zip(actual, shape):
            if expected == "N":
                if not isinstance(value, str):
                    raise BuildError(f"{component} requires dynamic batch")
            elif expected is not None and value != expected:
                raise BuildError(f"Wrong {component}/{role} IO dimensions")
    def named(items, names):
        if [item["name"] for item in items] != names:
            raise BuildError(f"Wrong {component}/{role} IO names")
    if component == "person":
        named(inputs, ["images", "orig_target_sizes"])
        named(outputs, ["labels", "boxes", "scores"])
        for item, dtype, shape in [(inputs[0], "float32", ["N", 3, 640, 640]),
                                   (inputs[1], "int64", ["N", 2]),
                                   (outputs[0], "int64", ["N", 300]),
                                   (outputs[1], "float32", ["N", 300, 4]),
                                   (outputs[2], "float32", ["N", 300])]:
            check(item, dtype, shape)
    elif component in ("body", "face_recognition"):
        if len(inputs) != 1 or len(outputs) != 1:
            raise BuildError(f"Wrong {component} IO count")
        height, width = (256, 128) if component == "body" else (112, 112)
        check(inputs[0], "float32", ["N", 3, height, width])
        check(outputs[0], "float32", ["N", 512])
    elif component == "face_detection":
        if len(inputs) != 1 or len(outputs) != 12:
            raise BuildError("Wrong YuNet IO count")
        check(inputs[0], "float32", [1, 3, "N", "N"])
        expected = {f"{head}_{stride}": channels for head, channels in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)) for stride in (8, 16, 32)}
        if set(item["name"] for item in outputs) != set(expected):
            raise BuildError("Wrong YuNet named heads")
        for item in outputs:
            check(item, "float32", [1, None, expected[item["name"]]])
    elif component == "scene_detector":
        named(inputs, ["pixel_values"])
        named(outputs, ["logits", "boxes"])
        check(inputs[0], "float32", [1, 3, 960, 960])
        check(outputs[0], "float32", [1, 3600, len(_scene_vocabulary().VOCABULARY)])
        check(outputs[1], "float32", [1, 3600, 4])
    elif role == "encoder":
        named(inputs, ["images"])
        named(outputs, ["image_embeddings"])
        check(inputs[0], "float32", [1, 3, 1024, 1024])
        check(outputs[0], "float32", [1, 256, 64, 64])
    else:
        named(inputs, ["image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input", "orig_im_size"])
        named(outputs, ["masks", "iou_predictions", "low_res_masks"])
        shapes = [[1, 256, 64, 64], [1, "N", 2], [1, "N"], [1, 1, 256, 256], [1], [2]]
        for item, shape in zip(inputs, shapes):
            check(item, "float32", shape)
        for item, shape in zip(outputs, [[1, None, None, None], [1, None], [1, None, 256, 256]]):
            check(item, "float32", shape)


def _prepare_cuda_dlls():
    """Locate existing CUDA DLLs without reading application settings."""
    if os.name != "nt" or _DLL_HANDLES:
        return
    directories = []
    for prefix in (Path(sys.prefix), ROOT / ".venv", ROOT / ".model-build"):
        site = prefix / "Lib/site-packages"
        directories.extend(site.glob("nvidia/*/bin"))
        directories.extend(site.glob("nvidia/*/bin/*"))
        directories.extend(site.glob("nvidia/*/lib"))
        directories.append(site / "torch/lib")
    directories.append(ROOT / "bin")
    directories.append(ROOT / "bin/cuda")
    selected = []
    for directory in directories:
        if directory.is_dir() and directory not in selected:
            selected.append(directory)
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    os.environ["PATH"] = os.pathsep.join([*(str(path) for path in selected), os.environ.get("PATH", "")])


@contextmanager
def _session(path, device):
    """Own profiling until every check exits, including provider rejection."""
    if device == "cuda":
        _prepare_cuda_dlls()
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.enable_profiling = device == "cuda"
    options.profile_file_prefix = str(path.with_suffix(".profile"))
    if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise BuildError("CUDAExecutionProvider is unavailable; use --device cpu explicitly")
    providers = [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    report = {}
    profile = None
    try:
        if device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
            raise BuildError("ORT failed to initialize CUDA")
        yield session, report
    finally:
        failure = sys.exception()
        if options.enable_profiling:
            try:
                profile = Path(session.end_profiling())
            except Exception as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"ORT profiling cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}")
    # Validation can raise too, but the native profile writer is already closed.
    report.update(_finish_session(session, device, profile))


def _finish_session(session, device, profile):
    result = {"providers": session.get_providers(), "device": device}
    if device == "cuda":
        if profile is None:
            raise BuildError("CUDA profiling did not produce a report")
        events = json.loads(profile.read_text(encoding="utf-8"))
        heavy = [event["args"] for event in events if event.get("cat") == "Node"
                 and event.get("args", {}).get("provider")
                 and any(op in event["args"].get("op_name", "").lower() for op in ("conv", "gemm", "matmul", "attention"))]
        if not heavy or any(item["provider"] != "CUDAExecutionProvider" for item in heavy):
            raise BuildError("CUDA profile does not place all heavy inference nodes on CUDA")
        result.update(heavy_cuda_nodes=len(heavy), profile=profile.name)
    return result


def _run(session, feeds):
    import numpy as np
    result = session.run(None, feeds)
    if not result or any(not np.isfinite(item).all() for item in result):
        raise BuildError("ONNX inference returned empty/nonfinite outputs")
    return result


def _numpy(torch, module, feeds):
    import numpy as np
    with torch.inference_mode():
        result = module(*(torch.from_numpy(value) for value in feeds.values()))
    if isinstance(result, torch.Tensor):
        result = (result,)
    arrays = [value.detach().cpu().numpy() for value in result]
    if not arrays or any(not np.isfinite(value).all() for value in arrays):
        raise BuildError("Source inference returned empty/nonfinite outputs")
    return arrays


def _normalized(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if values.ndim != 2 or values.shape[1] != 512 or not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise BuildError("Invalid 512-dimensional embedding")
    return values / norms


def _cosine_delta(expected, actual, tolerance):
    import numpy as np
    if expected.shape != actual.shape:
        raise BuildError("Embedding output shape mismatch")
    delta = float(np.max(np.abs(1 - np.sum(_normalized(expected) * _normalized(actual), axis=1))))
    if delta > tolerance:
        raise BuildError(f"Embedding cosine delta {delta:g} exceeds {tolerance:g}")
    return delta


def _person_delta(expected, actual, transforms, thresholds):
    import numpy as np
    worst_coordinate = worst_score = 0.0
    if any(left.shape != right.shape for left, right in zip(expected, actual)):
        raise BuildError("D-FINE output shape mismatch")
    for index, (ratio, left, top) in enumerate(transforms):
        a = np.flatnonzero(expected[0][index] == 0)
        b = np.flatnonzero(actual[0][index] == 0)
        if len(a) != len(b):
            raise BuildError("D-FINE source/runtime person-count mismatch")
        if not len(a):
            continue
        # Compare all returned person candidates, not only confident successes.
        # Top-k order can differ under ties: find a one-to-one assignment
        # within both tolerances rather than comparing incidental row order.
        coordinates_a = (expected[1][index, a] - np.array([left, top, left, top])) / ratio
        coordinates_b = (actual[1][index, b] - np.array([left, top, left, top])) / ratio
        scores_a, scores_b = expected[2][index, a], actual[2][index, b]
        costs = np.max(np.abs(coordinates_a[:, None] - coordinates_b[None]), axis=2)
        costs = np.maximum(costs / thresholds["person_coordinate_pixels"],
                           np.abs(scores_a[:, None] - scores_b[None]) / thresholds["person_score_absolute"])
        # Exact bipartite feasibility within the requested tolerances. This
        # avoids requiring SciPy solely in the export-only environment.
        matches = {}
        def assign(row, seen):
            for column in np.argsort(costs[row]):
                column = int(column)
                if costs[row, column] > 1 or column in seen:
                    continue
                seen.add(column)
                if column not in matches or assign(matches[column], seen):
                    matches[column] = row
                    return True
            return False
        if any(not assign(row, set()) for row in range(len(a))):
            raise BuildError("D-FINE person box/score parity exceeds 1px/.01")
        for column, row in matches.items():
            worst_coordinate = max(worst_coordinate, float(np.max(np.abs(coordinates_a[row] - coordinates_b[column]))))
            worst_score = max(worst_score, float(abs(scores_a[row] - scores_b[column])))
    return {"coordinate_error_pixels": worst_coordinate, "score_error": worst_score,
            "person_candidates": int((expected[0] == 0).sum())}


def _reference_rgb(reference_image):
    import numpy as np
    from PIL import Image
    if reference_image is not None:
        path = Path(reference_image)
        if not path.is_file() or path.is_symlink():
            raise BuildError("Reference image must be a local regular file")
        with Image.open(path) as image:
            if image.width * image.height > 16_000_000:
                raise BuildError("Reference image exceeds 16 megapixels")
            return np.asarray(image.convert("RGB"))
    y, x = np.indices((360, 640))
    return np.stack(((x * 3 + y) % 256, (x + y * 5) % 256, ((x // 24 + y // 24) % 2) * 200 + 20), axis=2).astype(np.uint8)


def _images(rgb, count, purpose):
    import numpy as np
    from PIL import Image
    rows, transforms = [], []
    for index in range(count):
        frame = np.roll(rgb, index * 13, axis=1)
        if index % 2:
            frame = np.transpose(frame, (1, 0, 2))
        image = Image.fromarray(frame)
        if purpose == "person":
            ratio = min(640 / image.width, 640 / image.height)
            width, height = int(image.width * ratio), int(image.height * ratio)
            left, top = (640 - width) // 2, (640 - height) // 2
            canvas = Image.new("RGB", (640, 640), (0, 0, 0))
            canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (left, top))
            value = np.asarray(canvas, dtype=np.float32) / 255
            transforms.append((ratio, left, top))
        elif purpose == "body":
            value = np.asarray(image.resize((128, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255
            value = (value - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        elif purpose == "face":
            value = (np.asarray(image.resize((112, 112), Image.Resampling.BILINEAR), dtype=np.float32) - 127.5) / 127.5
        elif purpose == "scene_detector":
            side = max(image.size)
            square = Image.new("RGB", (side, side), (127, 127, 127))
            square.paste(image, (0, 0))
            value = np.asarray(square.resize((960, 960), Image.Resampling.BOX), dtype=np.float32) / 255
            value = (value - np.array([.48145466, .4578275, .40821073], np.float32)) / np.array([.26862954, .26130258, .27577711], np.float32)
        else:
            raise BuildError("Unknown preprocessing purpose")
        rows.append(value.transpose(2, 0, 1))
    return np.ascontiguousarray(rows, dtype=np.float32), transforms


def _check_person(torch, model, path, rgb, device, thresholds):
    import numpy as np
    _graph_io(path, "person")
    with _session(path, device) as (session, report):
        checks = []
        for count in (1, 4):
            images, transforms = _images(rgb, count, "person")
            feeds = {"images": images, "orig_target_sizes": np.tile(np.array([[640, 640]], dtype=np.int64), (count, 1))}
            expected, actual = _numpy(torch, model, feeds), _run(session, feeds)
            checks.append({"batch": count, **_person_delta(expected, actual, transforms, thresholds)})
        if not sum(check["person_candidates"] for check in checks):
            raise BuildError("No source person candidates for parity; supply --reference-image")
    return {"checks": checks, **report}


def _embedding_chunks(session, images, chunk_size=16):
    import numpy as np
    if not 1 <= chunk_size <= 16:
        raise BuildError("Embedding chunk size must be 1..16")
    name = session.get_inputs()[0].name
    if len(images) == 0:
        return np.empty((0, 512), dtype=np.float32)
    rows = [_normalized(_run(session, {name: images[start:start + chunk_size]})[0])
            for start in range(0, len(images), chunk_size)]
    result = np.concatenate(rows, axis=0)
    if result.shape != (len(images), 512):
        raise BuildError("Embedding batch truncated or reordered")
    return result


def _check_embedding(torch, source, path, component, rgb, device, tolerance):
    import numpy as np
    _graph_io(path, component)
    with _session(path, device) as (session, report), \
            (_session(source, "cpu") if component == "face_recognition" else nullcontext((None, {}))) as (original, _):
        images, _ = _images(rgb, 17, "face" if component == "face_recognition" else "body")
        checks = []
        for count in (1, 4, 17):
            actual = _embedding_chunks(session, images[:count])
            if original is not None:
                # Independent singleton reference also proves output order across
                # the 16+1 chunk boundary, rather than repeating the same slicing.
                expected = np.concatenate([_embedding_chunks(original, images[index:index + 1]) for index in range(count)])
            else:
                expected = np.concatenate([_numpy(torch, source, {"images": images[index:index + 1]})[0] for index in range(count)])
            checks.append({"batch": count, "cosine_delta": _cosine_delta(expected, actual, tolerance)})
    return {"checks": checks, **report}


def _check_yunet(path, rgb, device):
    import numpy as np
    from PIL import Image
    _graph_io(path, "face_detection")
    with _session(path, device) as (session, report):
        name = session.get_inputs()[0].name
        shapes = []
        for width, height in ((640, 640), (320, 640)):
            image = np.asarray(Image.fromarray(rgb).resize((width, height)), dtype=np.float32)
            feeds = {name: np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None])}
            result = dict(zip([item.name for item in session.get_outputs()], _run(session, feeds)))
            for stride in (8, 16, 32):
                for head, channels in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)):
                    if result[f"{head}_{stride}"].shape != (1, height // stride * (width // stride), channels):
                        raise BuildError("YuNet runtime head shape mismatch")
            shapes.append([1, 3, height, width])
    return {"input_shapes": shapes, **report}


def _check_scene(torch, encoder, decoder, encoder_path, decoder_path, rgb, thresholds):
    import numpy as np
    from PIL import Image
    _graph_io(encoder_path, "scene", "encoder")
    _graph_io(decoder_path, "scene", "decoder")
    with _session(encoder_path, "cpu") as (encoder_session, _), \
            _session(decoder_path, "cpu") as (decoder_session, _):
        image = Image.fromarray(rgb)
        ratio = 1024 / max(image.width, image.height)
        width, height = int(image.width * ratio + .5), int(image.height * ratio + .5)
        resized = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        resized = (resized - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array([58.395, 57.12, 57.375], dtype=np.float32)
        data = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
        data[0, :, :height, :width] = resized.transpose(2, 0, 1)
        expected_embedding = _numpy(torch, encoder, {"images": data})[0]
        actual_embedding = _run(encoder_session, {"images": data})[0]
        def compare(expected, actual):
            if len(expected) != len(actual):
                raise BuildError("MobileSAM output count mismatch")
            for left, right in zip(expected, actual):
                if left.shape != right.shape or not np.allclose(left, right, rtol=thresholds["scene_relative"], atol=thresholds["scene_absolute"]):
                    raise BuildError("MobileSAM source/ORT FP32 parity failed")
        compare([expected_embedding], [actual_embedding])
        for points in (2, 5):
            coords = np.array([[[width * .5, height * .5]] * (points - 1) + [[0., 0.]]], dtype=np.float32)
            feeds = {"image_embeddings": actual_embedding, "point_coords": coords,
                     "point_labels": np.array([[1.] * (points - 1) + [-1.]], dtype=np.float32),
                     "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                     "has_mask_input": np.zeros(1, dtype=np.float32),
                     "orig_im_size": np.array([image.height, image.width], dtype=np.float32)}
            source_feeds = dict(feeds, image_embeddings=expected_embedding)
            compare(_numpy(torch, decoder, source_feeds), _run(decoder_session, feeds))
    return {"providers": ["CPUExecutionProvider"], "prompt_counts": [2, 5], "fp32_parity": "PASS"}


def _check_scene_detector(torch, model, path, rgb, device, thresholds):
    import numpy as np
    _graph_io(path, "scene_detector")
    images, _ = _images(rgb, 1, "scene_detector")
    feeds = {"pixel_values": images}
    expected = _numpy(torch, model, feeds)
    with _session(path, device) as (session, report):
        actual = _run(session, feeds)
        if len(expected) != 2 or len(actual) != 2:
            raise BuildError("OWLv2 output count mismatch")
        errors = {}
        for name, left, right, limit in zip(
                ("logit_absolute_error", "box_absolute_error"), expected, actual,
                (thresholds["scene_detector_logit_absolute"], thresholds["scene_detector_box_absolute"])):
            if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                raise BuildError("OWLv2 output shape/nonfinite mismatch")
            errors[name] = float(np.max(np.abs(left - right)))
            if errors[name] > limit:
                raise BuildError(f"OWLv2 {name} {errors[name]:g} exceeds {limit:g}")
    return {"fp32_parity": "PASS", **errors, **report}


def _try_fp16(path, component, checker, device):
    """A failed candidate never replaces the verified, full FP32 network."""
    if device != "cuda":
        return path, {"status": "NOT_ATTEMPTED", "reason": "CUDA validation required", "active_precision": "FP32",
                      "capacity_implication": "FP32 memory/throughput; capacity must be measured"}
    candidate = path.with_name(path.stem.replace("fp32", "fp16") + path.suffix)
    try:
        import onnx
        from onnxconverter_common import float16
        model = onnx.load(str(path))
        model = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=False)
        onnx.save(model, str(candidate))
        result = checker(candidate)
        return candidate, {"status": "PASS", "active_precision": "FP16", "sha256": sha256_file(candidate), "validation": result}
    except Exception as exc:
        candidate.unlink(missing_ok=True)
        reason = "\n".join([f"{type(exc).__name__}: {exc}", *getattr(exc, "__notes__", ())])
        return path, {"status": "REJECTED", "active_precision": "FP32", "reason": reason,
                      "capacity_implication": "Full FP32 fallback; lower capacity possible, benchmark required"}


def _repair_auraface_batch(source: Path, destination: Path):
    """Correct the source's stale output batch annotation; never alter computation."""
    import onnx
    model = onnx.load(str(source), load_external_data=False)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise BuildError("AuraFace requires one input and output")
    inputs = model.graph.input[0].type.tensor_type.shape.dim
    outputs = model.graph.output[0].type.tensor_type.shape.dim
    if (len(inputs) != 4 or not inputs[0].dim_param or len(outputs) != 2
            or outputs[1].dim_value != 512
            or not (outputs[0].dim_param or outputs[0].dim_value == 1)):
        raise BuildError("Unexpected AuraFace batch declaration")
    outputs[0].Clear()
    outputs[0].dim_param = inputs[0].dim_param
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(destination))


def _file_metadata(stage, path, component, role, lock, exporter_hash):
    artifact = lock["artifacts"][component]
    inputs, outputs = _graph_io(path, component, role)
    converted = "source" in artifact or component in ("face_recognition", "scene_detector")
    upstream = lock["sources"][artifact["source"]]["revision"] if "source" in artifact else artifact["revision"]
    return {"role": role, "path": path.relative_to(stage).as_posix(), "sha256": sha256_file(path),
            "source_url": artifact["url"], "source_revision": artifact["revision"],
            "source_sha256": artifact["sha256"], "license": artifact["license"],
            "license_url": artifact["license_url"],
            "exporter_revision": f"{upstream}:{exporter_hash}" if converted else f"native:{upstream}",
            "inputs": inputs, "outputs": outputs}


def _build_components(stage, cache, device, reference_image):
    # Heavy dependencies are loaded here only, after the offline CLI was invoked.
    import numpy as np
    import torch
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["version"] != 1 or lock["alignment_id"] != ALIGNMENT_ID or lock["opset"] != 16:
        raise BuildError("Unsupported build lock")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    np.random.seed(0)
    started = time.monotonic()
    sources = {name: download_verified(spec, cache) for name, spec in lock["sources"].items()}
    artifacts = {name: download_verified(spec, cache) for name, spec in lock["artifacts"].items()}
    notices = stage / "notices"
    notices.mkdir()
    for name, spec in lock["notices"].items():
        shutil.copyfile(download_verified(spec, cache), notices / name)
    # Keep the entire lock and exporter, not just filenames/mtimes, for audit.
    shutil.copyfile(LOCK_PATH, stage / "model_build_lock.json")
    shutil.copyfile(Path(__file__), stage / "prepare_models.py")
    shutil.copyfile(Path(__file__).with_name("model_build_requirements.txt"), stage / "model_build_requirements.txt")
    exporter_hash = sha256_file(Path(__file__))
    report = {"version": 1, "requested_device": device, "exporter_sha256": exporter_hash,
              "lock_sha256": sha256_file(LOCK_PATH), "capacity": "NOT_EVALUATED",
              "reference": {"kind": "local_image", "sha256": sha256_file(Path(reference_image))} if reference_image else {"kind": "DETERMINISTIC_SYNTHETIC"},
              "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "timm", "onnx", "onnxruntime-gpu", "onnxconverter-common", "numpy", "Pillow", "transformers")},
              "components": {}}
    rgb = _reference_rgb(reference_image)
    thresholds = lock["parity"]
    with tempfile.TemporaryDirectory(prefix="source-", dir=cache) as work:
        work = Path(work)
        roots = {name: extract_source(archive, work / name) for name, archive in sources.items()}
        auxiliary = work / "owlv2-auxiliary"
        auxiliary.mkdir()
        for name, spec in lock["auxiliary"]["scene_detector"].items():
            shutil.copyfile(download_verified(spec, cache), auxiliary / name)
        for name, root in roots.items():
            license_path = root / "LICENSE"
            if not license_path.is_file():
                raise BuildError(f"Missing upstream license for {name}")
            shutil.copyfile(license_path, notices / f"{name}-LICENSE")
            for extra in root.glob("NOTICE*"):
                if extra.is_file():
                    shutil.copyfile(extra, notices / f"{name}-{extra.name}")
        person_path = stage / "dfine_s_coco_fp32.onnx"
        person = _dfine(torch, roots["dfine"], artifacts["person"], person_path)
        person_check = _check_person(torch, person, person_path, rgb, device, thresholds)
        person_active, candidate = _try_fp16(person_path, "person", lambda path: _check_person(torch, person, path, rgb, "cuda", thresholds), device)
        report["components"]["person"] = {"source_sha256": lock["artifacts"]["person"]["sha256"], "fp32": person_check, "fp16": candidate}
        del person
        body_path = stage / "osnet_ain_x1_0_fp32.onnx"
        body = _osnet(torch, roots["osnet"], artifacts["body"], body_path)
        body_check = _check_embedding(torch, body, body_path, "body", rgb, device, thresholds["body_cosine_delta"])
        body_active, candidate = _try_fp16(body_path, "body", lambda path: _check_embedding(torch, body, path, "body", rgb, "cuda", thresholds["body_fp16_cosine_delta"]), device)
        report["components"]["body"] = {"source_sha256": lock["artifacts"]["body"]["sha256"], "fp32": body_check, "fp16": candidate}
        del body
        encoder_path, decoder_path = stage / "mobile_sam_encoder.onnx", stage / "mobile_sam_decoder.onnx"
        encoder, decoder = _mobilesam(torch, roots["mobilesam"], artifacts["scene"], encoder_path, decoder_path)
        report["components"]["scene"] = _check_scene(torch, encoder, decoder, encoder_path, decoder_path, rgb, thresholds)
        del encoder, decoder
        scene_detector_path = stage / "owlv2_scene_detector_fp32.onnx"
        scene_detector = _owlv2(torch, auxiliary, artifacts["scene_detector"], scene_detector_path)
        report["components"]["scene_detector"] = _check_scene_detector(
            torch, scene_detector, scene_detector_path, rgb, device, thresholds)
        del scene_detector
    face_detection = stage / "face_detection_yunet_2026may.onnx"
    face_recognition = stage / "glintr100.onnx"
    shutil.copyfile(artifacts["face_detection"], face_detection)
    _repair_auraface_batch(artifacts["face_recognition"], face_recognition)
    report["components"]["face_detection"] = _check_yunet(face_detection, rgb, device)
    report["components"]["face_recognition"] = _check_embedding(torch, artifacts["face_recognition"], face_recognition, "face_recognition", rgb, device, thresholds["recognition_cosine_delta"])
    report["components"]["face_recognition"]["preparation"] = "DYNAMIC_OUTPUT_BATCH_DECLARATION_V1"
    paths = {"person": [("model", person_active)], "face_detection": [("model", face_detection)],
             "face_recognition": [("model", face_recognition)], "body": [("model", body_active)],
             "scene": [("encoder", encoder_path), ("decoder", decoder_path)],
             "scene_detector": [("model", scene_detector_path)]}
    manifest = {"version": 1, "alignment_id": ALIGNMENT_ID, "components": {}}
    for component, files in paths.items():
        manifest["components"][component] = {"preprocessing_id": lock["artifacts"][component]["preprocessing_id"],
            "files": [_file_metadata(stage, path, component, role, lock, exporter_hash) for role, path in files]}
    report["elapsed_seconds"] = time.monotonic() - started
    report["notices"] = {path.name: sha256_file(path) for path in notices.iterdir()}
    # An approved FP16 manifest declares only the active candidate; keep FP32
    # hash/parity in the report, not a second runtime-model declaration.
    for component, original, active in (("person", person_path, person_active), ("body", body_path, body_active)):
        report["components"][component]["fp32_sha256"] = sha256_file(original)
        report["components"][component]["active_sha256"] = sha256_file(active)
        if active != original:
            original.unlink()
    (stage / "build-report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def prepare_bundle(output: Path, cache: Path, device: str, reference_image: Path | None = None) -> Path:
    if device not in ("cpu", "cuda"):
        raise BuildError("Device must be cpu or cuda")
    output, cache = Path(output).absolute(), Path(cache).absolute()
    if cache == output or cache.is_relative_to(output) or output.is_relative_to(cache):
        raise BuildError("Cache and published bundle must not overlap")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".model-stage-", dir=output.parent))
    try:
        manifest = _build_components(stage, cache, device, reference_image)
        return publish_bundle(stage, output, manifest, uuid.uuid4().hex)
    finally:
        failure = sys.exception()
        if stage.exists():
            try:
                shutil.rmtree(stage)
            except OSError as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"Staging cleanup failed for {stage}: {type(cleanup_error).__name__}: {cleanup_error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/store-v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cache", type=Path, default=ROOT / ".model-build/cache")
    parser.add_argument("--reference-image", type=Path, help="Optional authorized local image; never copied into the bundle")
    args = parser.parse_args(argv)
    try:
        report = prepare_bundle(args.output, args.cache, args.device, args.reference_image)
    except Exception as exc:
        print(f"MODEL_BUILD_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(note, file=sys.stderr)
        return 1
    print(f"Manifest: {args.output / 'manifest.json'}")
    print(f"Validation report: {report}")
    print("Capacity is NOT_EVALUATED; FP32 fallbacks may reduce measured capacity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
