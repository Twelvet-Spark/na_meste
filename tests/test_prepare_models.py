"""Build boundaries; real torch/ORT regressions run only in a build-env subprocess.

Run: .venv/Scripts/python.exe tests/test_prepare_models.py
The actual model/parity proof is tools/prepare_models.py --device cpu|cuda.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("prepare_models", ROOT / "tools/prepare_models.py")
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def artifact(data=b"approved artifact"):
    return {"url": "https://raw.githubusercontent.com/approved/project/" + "a" * 40 + "/model",
            "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def embedding_model(path, *, negate=False):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    weights = np.zeros((3, 512), dtype=np.float32)
    weights[:, :3] = np.eye(3, dtype=np.float32) * (-1 if negate else 1)
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["images"], ["mean"]),
         helper.make_node("Flatten", ["mean"], ["flat"], axis=1),
         helper.make_node("MatMul", ["flat", "weights"], ["embedding"])],
        "profile-lifecycle",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 112, 112])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["N", 512])],
        [numpy_helper.from_array(weights, "weights")])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=10), path)


@contextmanager
def profiled_cpu_sessions():
    """Keep real sessions alive so GC cannot conceal missing profile closure."""
    import onnxruntime as ort
    constructor = ort.InferenceSession
    sessions = []
    def cpu_session(path, *, sess_options, providers):
        sess_options.enable_profiling = True
        session = constructor(path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        sessions.append(session)
        return session
    with patch.object(ort, "InferenceSession", side_effect=cpu_session), \
            patch.object(ort, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]), \
            patch.object(prepare, "_prepare_cuda_dlls"):
        try:
            yield sessions
        finally:
            # Test-only safety after assertions: clean up even on the old,
            # leaking implementation, without obscuring the assertion failure.
            for session in sessions:
                session.end_profiling()


class PrepareBoundaryTests(unittest.TestCase):
    def assert_profiles_closed(self, root, count):
        profiles = list(root.glob("*.profile*.json"))
        self.assertEqual(len(profiles), count)
        for profile in profiles:
            self.assertIsInstance(json.loads(profile.read_text(encoding="utf-8")), list)
            renamed = profile.with_suffix(".closed")
            profile.rename(renamed)  # Windows refuses this if ORT still owns the open profile.
            renamed.unlink()

    def test_cuda_dll_discovery_includes_nested_wheel_and_project_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / ".venv/Lib/site-packages/nvidia/cu13/bin/x86_64"
            flat = root / ".model-build/Lib/site-packages/nvidia/cudnn/bin"
            project = root / "bin/cuda"
            for path in (nested, flat, project):
                path.mkdir(parents=True)
            (nested / "cublasLt64_13.dll").write_bytes(b"discovery fixture, never loaded")
            registered = []
            def register(path):
                registered.append(Path(path))
                return object()
            environment = {"PATH": "existing-search-path"}
            fake_os = SimpleNamespace(name="nt", add_dll_directory=register,
                                      environ=environment, pathsep=prepare.os.pathsep)
            with patch.object(prepare, "ROOT", root), patch.object(prepare.sys, "prefix", str(root / ".venv")), \
                    patch.object(prepare, "os", fake_os), patch.object(prepare, "_DLL_HANDLES", []):
                prepare._prepare_cuda_dlls()
                prepare._prepare_cuda_dlls()
                for path in (nested, flat, project):
                    self.assertEqual(registered.count(path), 1)
                    self.assertIn(str(path), environment["PATH"].split(fake_os.pathsep))
                self.assertEqual(len(prepare._DLL_HANDLES), len(registered))

    def test_cuda_fallback_closes_profile_before_failed_build_cleanup(self):
        with tempfile.TemporaryDirectory() as directory, profiled_cpu_sessions() as sessions:
            root = Path(directory)
            output = root / "bundle"
            output.mkdir()
            (output / "manifest.json").write_bytes(b"previous complete bundle")
            def build(stage, *args):
                path = stage / "fallback.onnx"
                embedding_model(path)
                with prepare._session(path, "cuda"):
                    self.fail("CPU fallback was admitted as CUDA")
            remove = prepare.shutil.rmtree
            def cleanup(path, *args, **kwargs):
                if Path(path).name.startswith(".model-stage-"):
                    self.assertEqual(len(sessions), 1)
                    self.assert_profiles_closed(Path(path), 1)
                return remove(path, *args, **kwargs)
            with patch.object(prepare, "_build_components", side_effect=build), \
                    patch.object(prepare.shutil, "rmtree", side_effect=cleanup):
                with self.assertRaisesRegex(prepare.BuildError, "ORT failed to initialize CUDA"):
                    prepare.prepare_bundle(output, root / "cache", "cuda")
            self.assertEqual((output / "manifest.json").read_bytes(), b"previous complete bundle")
            self.assertEqual(list(root.glob(".model-stage-*")), [])

    def test_real_inference_failure_closes_profile_and_preserves_exception(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory, profiled_cpu_sessions() as sessions:
            root = Path(directory)
            path = root / "inference.onnx"
            embedding_model(path)
            failure = None
            try:
                with prepare._session(path, "cpu") as (session, report):
                    try:
                        prepare._run(session, {"images": np.ones((1, 3, 112, 112), dtype=np.int64)})
                    except Exception as exc:
                        failure = exc
                        raise
            except Exception as exc:
                self.assertIsNotNone(failure)
                self.assertIs(exc, failure)
            else:
                self.fail("Invalid inference input was accepted")
            self.assertEqual(len(sessions), 1)
            self.assert_profiles_closed(root, 1)

    def test_embedding_parity_failure_closes_both_real_profiles(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory, profiled_cpu_sessions() as sessions:
            root = Path(directory)
            source, target = root / "source.onnx", root / "target.onnx"
            embedding_model(source)
            embedding_model(target, negate=True)
            with self.assertRaisesRegex(prepare.BuildError, "Embedding cosine delta"):
                prepare._check_embedding(None, source, target, "face_recognition",
                                         np.zeros((8, 8, 3), dtype=np.uint8), "cpu", 1e-4)
            self.assertEqual(len(sessions), 2)
            self.assert_profiles_closed(root, 2)

    def test_closed_real_cpu_profile_cannot_certify_cuda_heavy_nodes(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory, profiled_cpu_sessions():
            root = Path(directory)
            path = root / "success.onnx"
            embedding_model(path)
            with prepare._session(path, "cpu") as (session, report):
                prepare._run(session, {"images": np.ones((1, 3, 112, 112), dtype=np.float32)})
            self.assertEqual(report, {"providers": ["CPUExecutionProvider"], "device": "cpu"})
            profiles = list(root.glob("*.profile*.json"))
            self.assertEqual(len(profiles), 1)
            events = json.loads(profiles[0].read_text(encoding="utf-8"))
            self.assertTrue(any(event.get("args", {}).get("op_name") == "MatMul" for event in events))
            with self.assertRaisesRegex(prepare.BuildError, "heavy inference nodes"):
                prepare._finish_session(session, "cuda", profiles[0])
            self.assert_profiles_closed(root, 1)

    def test_profile_cleanup_failure_keeps_primary_error_visible(self):
        import onnxruntime as ort
        session_class = ort.InferenceSession
        end = session_class.end_profiling
        def close_then_fail(session):
            end(session)
            raise OSError("profile close failure")
        with tempfile.TemporaryDirectory() as directory, profiled_cpu_sessions():
            path = Path(directory) / "cleanup.onnx"
            embedding_model(path)
            failure = prepare.BuildError("original parity failure")
            with self.assertRaises(prepare.BuildError) as raised:
                with patch.object(session_class, "end_profiling", close_then_fail):
                    with prepare._session(path, "cpu"):
                        raise failure
            self.assertIs(raised.exception, failure)
            self.assertTrue(any("profile close failure" in note for note in raised.exception.__notes__))
            self.assert_profiles_closed(Path(directory), 1)

    def test_staging_cleanup_failure_keeps_build_error_and_reports_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failure = prepare.BuildError("original build failure")
            with patch.object(prepare, "_build_components", side_effect=failure), \
                    patch.object(prepare.shutil, "rmtree", side_effect=PermissionError("stage still locked")):
                with self.assertRaises(prepare.BuildError) as raised:
                    prepare.prepare_bundle(root / "bundle", root / "cache", "cpu")
            self.assertIs(raised.exception, failure)
            self.assertTrue(any("stage still locked" in note for note in raised.exception.__notes__))

    def test_auraface_output_batch_repair_preserves_ordered_vectors_and_source(self):
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.onnx"
            output = Path(directory) / "prepared.onnx"
            weights = np.zeros((3, 512), dtype=np.float32)
            weights[:, :3] = np.eye(3, dtype=np.float32)
            graph = helper.make_graph(
                [helper.make_node("GlobalAveragePool", ["data"], ["mean"]),
                 helper.make_node("Flatten", ["mean"], ["flat"], axis=1),
                 helper.make_node("MatMul", ["flat", "weights"], ["embedding"])],
                "dynamic-input-stale-output-declaration",
                [helper.make_tensor_value_info("data", TensorProto.FLOAT, ["N", 3, 112, 112])],
                [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 512])],
                [numpy_helper.from_array(weights, "weights")])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=10)
            onnx.save(model, source)
            original = source.read_bytes()
            prepare._repair_auraface_batch(source, output)
            self.assertEqual(source.read_bytes(), original)
            repaired = onnx.load(output)
            self.assertEqual(repaired.graph.initializer, model.graph.initializer)
            self.assertEqual(repaired.graph.node, model.graph.node)
            with prepare._session(output, "cpu") as (session, _):
                images = np.ones((17, 3, 112, 112), dtype=np.float32)
                images[:, 0] = np.arange(1, 18, dtype=np.float32)[:, None, None]
                actual = prepare._embedding_chunks(session, images)
                expected = np.zeros((17, 512), dtype=np.float32)
                expected[:, :3] = np.column_stack((np.arange(1, 18), np.ones(17), np.ones(17)))
                expected /= np.linalg.norm(expected, axis=1, keepdims=True)
                np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_verified_cache_reused_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            spec = artifact()
            cached = cache / spec["sha256"]
            cached.write_bytes(b"approved artifact")
            with patch.object(prepare.urllib.request, "urlopen", side_effect=AssertionError("network")):
                self.assertEqual(prepare.download_verified(spec, cache), cached)

    def test_corrupt_cache_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            spec = artifact()
            cached = cache / spec["sha256"]
            cached.write_bytes(b"corrupt")
            with patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(b"approved artifact")):
                self.assertEqual(prepare.download_verified(spec, cache).read_bytes(), b"approved artifact")

    def test_bad_download_cannot_replace_cached_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            spec = artifact()
            cached = cache / spec["sha256"]
            cached.write_bytes(b"existing corrupt bytes")
            with patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaises(prepare.BuildError):
                    prepare.download_verified(spec, cache)
            self.assertEqual(cached.read_bytes(), b"existing corrupt bytes")
            self.assertEqual(list(cache.glob("*.part")), [])

    def test_interrupted_download_leaves_no_usable_artifact(self):
        class Interrupted(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise OSError("connection lost")
                return super().read(4)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch.object(prepare.urllib.request, "urlopen", return_value=Interrupted(b"approved artifact")):
                with self.assertRaises(prepare.BuildError):
                    prepare.download_verified(artifact(), cache)
            self.assertEqual(list(cache.iterdir()), [])

    def test_declared_size_and_git_blob_are_verified(self):
        for change in ({"size": 1}, {"git_blob_sha1": "0" * 40}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                spec = artifact()
                spec.update(change)
                with patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(b"approved artifact")):
                    with self.assertRaises(prepare.BuildError):
                        prepare.download_verified(spec, Path(directory))

    def test_non_https_sources_are_rejected_before_network(self):
        for url in ("http://example.com/model", "file:///tmp/model", "https://untrusted.example/model"):
            with self.subTest(url=url), tempfile.TemporaryDirectory() as directory:
                spec = artifact()
                spec["url"] = url
                with patch.object(prepare.urllib.request, "urlopen", side_effect=AssertionError("network")):
                    with self.assertRaises(prepare.BuildError):
                        prepare.download_verified(spec, Path(directory))

    def test_archive_traversal_and_symlinks_are_rejected(self):
        for member in ("root/../../escape", "/absolute", "root/link"):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / "source.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    info = zipfile.ZipInfo(member)
                    if member == "root/link":
                        info.create_system = 3
                        info.external_attr = 0o120777 << 16
                    stream.writestr(info, "target")
                with self.assertRaises(prepare.BuildError):
                    prepare.extract_source(archive, root / "source")
                self.assertFalse((root / "escape").exists())

    def test_publish_switches_manifest_and_retains_previous_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle"
            for identifier, data in (("first", b"first"), ("second", b"second")):
                stage = root / identifier
                stage.mkdir()
                (stage / "model.onnx").write_bytes(data)
                (stage / "build-report.json").write_text('{"verified":true}')
                manifest = {"components": {"person": {"files": [{"path": "model.onnx", "sha256": hashlib.sha256(data).hexdigest()}]}}}
                prepare.publish_bundle(stage, output, manifest, identifier)
                loaded = json.loads((output / "manifest.json").read_text())
                path = loaded["components"]["person"]["files"][0]["path"]
                self.assertEqual((output / path).read_bytes(), data)
            self.assertEqual((output / "generations/first/model.onnx").read_bytes(), b"first")
            self.assertTrue((output / "generations/second/build-report.json").is_file())

    def test_failed_publication_preserves_complete_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle"
            output.mkdir()
            old = b'{"old":"complete"}'
            (output / "manifest.json").write_bytes(old)
            stage = root / "stage"
            stage.mkdir()
            (stage / "model.onnx").write_bytes(b"new")
            manifest = {"components": {"person": {"files": [{"path": "model.onnx", "sha256": hashlib.sha256(b"new").hexdigest()}]}}}
            real_replace = prepare.os.replace
            def fail_commit(source, target):
                if Path(target) == output / "manifest.json":
                    raise OSError("publication blocked")
                return real_replace(source, target)
            with patch.object(prepare.os, "replace", side_effect=fail_commit):
                with self.assertRaises(OSError):
                    prepare.publish_bundle(stage, output, manifest, "new")
            self.assertEqual((output / "manifest.json").read_bytes(), old)
            self.assertFalse((output / "generations/new").exists())

    def test_publish_rejects_missing_or_changed_artifact(self):
        for body in (None, b"changed"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                stage = root / "stage"
                stage.mkdir()
                if body is not None:
                    (stage / "model.onnx").write_bytes(body)
                manifest = {"components": {"person": {"files": [{"path": "model.onnx", "sha256": "0" * 64}]}}}
                with self.assertRaises(prepare.BuildError):
                    prepare.publish_bundle(stage, root / "output", manifest, "new")
                self.assertFalse((root / "output/manifest.json").exists())

    def test_publish_rejects_escape_and_does_not_mutate_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage"
            stage.mkdir()
            manifest = {"components": {"person": {"files": [{"path": "../outside", "sha256": "0" * 64}]}}}
            original = deepcopy(manifest)
            with self.assertRaises(prepare.BuildError):
                prepare.publish_bundle(stage, root / "output", manifest, "new")
            self.assertEqual(manifest, original)

    def test_export_failure_does_not_touch_existing_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle"
            output.mkdir()
            (output / "manifest.json").write_bytes(b"complete")
            with patch.object(prepare, "_build_components", side_effect=prepare.BuildError("export failed")):
                with self.assertRaises(prepare.BuildError):
                    prepare.prepare_bundle(output, root / "cache", "cpu")
            self.assertEqual((output / "manifest.json").read_bytes(), b"complete")
            self.assertEqual(list(root.glob(".model-stage-*")), [])

    def test_nonfinite_state_requires_exact_architecture_mask(self):
        python = ROOT / ".model-build" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if not python.is_file():
            self.skipTest("Isolated .model-build Python absent; real torch guard requires build environment")
        code = r'''
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location("prepare_models", Path("tools/prepare_models.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

class Architecture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.decoder = torch.nn.Module()
        self.decoder.register_buffer("anchors", torch.tensor([[[0.25, 0.5, float("inf"), float("inf")]]]))
        self.register_buffer("finite_buffer", torch.ones(2))

model = Architecture()
references = dict(model.named_buffers())
state = model.state_dict()
checked = prepare._state_dict(torch, state, buffers=references)
assert torch.equal(checked["decoder.anchors"], references["decoder.anchors"])
assert torch.isposinf(checked["decoder.anchors"]).sum().item() == 2
assert torch.equal(checked["weight"], model.weight)
prefixed = {"module." + name: value for name, value in state.items()}
assert torch.equal(prepare._state_dict(torch, prefixed, buffers=references)["decoder.anchors"], state["decoder.anchors"])

def reject(changes, allowed=references):
    candidate = dict(state)
    candidate.update(changes)
    try:
        prepare._state_dict(torch, candidate, buffers=allowed)
    except prepare.BuildError:
        return
    raise AssertionError("Invalid/nonfinite checkpoint state accepted")

anchors = state["decoder.anchors"]
altered = anchors.clone()
altered[0, 0, 0] = 0.75
candidate = dict(state, **{"decoder.anchors": altered})
preserved = altered.clone()
checked = prepare._state_dict(torch, candidate, buffers=references)
assert torch.equal(checked["decoder.anchors"], preserved)  # Finite checkpoint values remain authoritative.
assert torch.equal(altered, preserved)  # No clamping or replacement.
moved = anchors.clone()
moved[0, 0, 1], moved[0, 0, 2] = float("inf"), 0.5
reject({"decoder.anchors": moved})  # Same infinity count, different positions.
reject({"decoder.anchors": torch.tensor([[[0.25, 0.5, 0.1, 0.2]]])})  # Removing all sentinels is invalid too.
reject({"decoder.anchors": -anchors})  # Infinity signs must match.
reject({"decoder.anchors": anchors.to(torch.float64)})
reject({"decoder.anchors": anchors.flatten()})
reject({"decoder.anchors": torch.full_like(anchors, float("nan"))})
reject({"decoder.anchors": anchors}, allowed={})  # No name-specific exception.
reject({"finite_buffer": torch.tensor([1., float("inf")])})
reject({"weight": torch.tensor([1., float("inf")])})
reject({"weight": torch.tensor([1., float("nan")])})
# An architecture's NaN buffer is not a permissible structural sentinel.
nan = torch.tensor([float("nan")])
reject({"decoder.anchors": nan}, allowed={"decoder.anchors": nan})
assert torch.equal(model.decoder.anchors, anchors)  # No sanitization or mutation.
print("Real torch structural-buffer guard passed")
'''
        result = subprocess.run([str(python), "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_dfine_export_traces_runtime_anchor_batch(self):
        python = ROOT / ".model-build" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if not python.is_file():
            self.skipTest("Isolated .model-build Python absent; real torch/ORT export requires build environment")
        # Reverting the production export sample to N=1 must fail real N=4
        # inference here, even if every graph IO axis still declares dynamic N.
        code = r'''
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("prepare_models", Path("tools/prepare_models.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
torch.set_num_threads(2)

class ConditionalAnchors(torch.nn.Module):
    """Tiny real graph retaining the upstream shape-dependent repeat branch."""
    def __init__(self):
        super().__init__()
        self.register_buffer("anchors", torch.arange(1200, dtype=torch.float32).reshape(1, 300, 4))

    def forward(self, images, orig_target_sizes):
        anchors = self.anchors
        if images.shape[0] > 1:
            anchors = anchors.repeat(images.shape[0], 1, 1)
        # Flattened per-image indices require the repeated anchor rows. The
        # N=1 trace omits repeat, so later batch rows index beyond the buffer.
        rows = torch.arange(images.shape[0]).unsqueeze(1) * 300 + torch.arange(300)
        indices = rows.reshape(-1, 1).expand(-1, 4)
        boxes = torch.gather(anchors.reshape(-1, 4), 0, indices).reshape(-1, 300, 4)
        boxes = boxes + orig_target_sizes[:, 0].to(torch.float32).reshape(-1, 1, 1)
        scores = images[:, 0, 0, 0].unsqueeze(1).expand(-1, 300)
        labels = torch.zeros_like(scores, dtype=torch.int64)
        return labels, boxes, scores

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "conditional-anchors.onnx"
    model = ConditionalAnchors().eval()
    prepare._export_dfine(torch, model, path)
    prepare._graph_io(path, "person")
    with prepare._session(path, "cpu") as (session, _):
        # Exercise smaller and larger batches than the trace sample: N=5 also
        # rejects a repeat factor accidentally hardcoded to four.
        for count in (1, 2, 4, 5):
            images = np.zeros((count, 3, 640, 640), dtype=np.float32)
            images[:, 0, 0, 0] = np.arange(count, dtype=np.float32) + 0.25
            sizes = np.full((count, 2), 640, dtype=np.int64)
            sizes[:, 0] += np.arange(count)
            feeds = {"images": images, "orig_target_sizes": sizes}
            actual = prepare._run(session, feeds)
            expected = [
                np.zeros((count, 300), dtype=np.int64),
                np.stack([np.arange(1200, dtype=np.float32).reshape(300, 4) + 640 + row
                          for row in range(count)]),
                np.stack([np.full(300, row + 0.25, dtype=np.float32) for row in range(count)]),
            ]
            source = prepare._numpy(torch, model, feeds)
            for wanted, native, exported in zip(expected, source, actual, strict=True):
                np.testing.assert_array_equal(native, wanted)
                np.testing.assert_array_equal(exported, wanted)
print("Real torch/ORT conditional-anchor export passed at N=1/N=2/N=4/N=5")
'''
        result = subprocess.run([str(python), "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        # Keep exporter warnings visible on success as well as on failure.
        if result.stderr:
            sys.stderr.write(result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
