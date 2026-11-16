"""CUDA DLL discovery/search contracts without ORT, hardware or native DLL loads."""
import glob
import importlib.util
import ntpath
import os
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

TESTS = []
SOURCE = Path(__file__).resolve().parent.parent / "app" / "ai" / "providers.py"


def test(fn):
    TESTS.append(fn)
    return fn


def subject():
    # Each case owns any retained DLL-directory handles; do not change app state.
    spec = importlib.util.spec_from_file_location("providers_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key(path):
    return ntpath.normcase(ntpath.normpath(path))


class DirectoryHandle:
    def __init__(self, search, directory):
        self.search = search
        self.directory = directory
        self.closed = False
        search.registered[directory] = search.registered.get(directory, 0) + 1

    def close(self):
        if not self.closed:
            self.closed = True
            self.search.registered[self.directory] -= 1
            if not self.search.registered[self.directory]:
                del self.search.registered[self.directory]

    def __del__(self):
        self.close()


class WindowsSearch:
    """Only replace filesystem/native boundaries, never the provider's PATH logic.

    CPython 3.8+ ctypes defaults search the DLL's directory and registered user
    directories, NOT PATH. Retaining AddDllDirectory's handle keeps a directory
    available after setup; closing/releasing it removes the search location.
    """
    def __init__(self, directories, path):
        self.directories = {key(directory) for directory in directories}
        self.files = {
            key(ntpath.join(directory, name))
            for directory, names in directories.items() for name in names
        }
        self.environ = {"PATH": path}
        self.registered = {}
        self.loaded = {}
        self.probe = None
        self.probe_results = []
        self.fail_loads = False

    def add_dll_directory(self, directory):
        if not ntpath.isabs(directory) or key(directory) not in self.directories:
            raise OSError("DLL directory must exist and be absolute")
        return DirectoryHandle(self, key(directory))

    def resolve(self, name, load_directory=None):
        if ntpath.isabs(name):
            return key(name) if key(name) in self.files else None
        if key(name) in self.loaded:
            return self.loaded[key(name)]
        directories = ([load_directory] if load_directory else []) + list(self.registered)
        for directory in directories:
            candidate = key(ntpath.join(directory, name))
            if candidate in self.files:
                return candidate
        return None

    def load(self, name):
        if self.probe:
            self.probe_results.append(self.resolve(self.probe, ntpath.dirname(name)))
        resolved = self.resolve(name)
        if self.fail_loads or resolved is None:
            raise OSError("unavailable or incompatible native dependency")
        self.loaded[key(ntpath.basename(name))] = resolved
        return SimpleNamespace()


@contextmanager
def windows_cuda(directories, candidates=None, path=""):
    module = subject()
    search = WindowsSearch(directories, path)
    paths = SimpleNamespace(
        join=ntpath.join, abspath=ntpath.abspath, normpath=ntpath.normpath,
        normcase=ntpath.normcase, isabs=ntpath.isabs,
        isdir=lambda directory: key(directory) in search.directories,
        isfile=lambda filename: key(filename) in search.files,
    )
    isolated_os = SimpleNamespace(
        name="nt", path=paths, pathsep=";", environ=search.environ,
        add_dll_directory=search.add_dll_directory,
    )
    with patch.object(module, "os", isolated_os), \
            patch.object(module, "ctypes", SimpleNamespace(WinDLL=search.load)), \
            patch.object(module, "_candidate_dirs", return_value=list(candidates or directories)):
        yield module, search


@test
def path_substrings_do_not_hide_a_required_directory():
    with windows_cuda({r"C:\cuda": ["cudart64_13.dll"]},
                      path=r"C:\cuda-tools;C:\Windows") as (module, search):
        module.setup_cuda_dlls()
        assert search.environ["PATH"] == r"C:\cuda;C:\cuda-tools;C:\Windows", search.environ


@test
def existing_path_membership_uses_normalized_complete_entries():
    original = r"C:/CUDA/.;C:\Windows"
    with windows_cuda({r"c:\cuda": ["cudart64_12.dll"]}, path=original) as (module, search):
        module.setup_cuda_dlls()
        assert search.environ["PATH"] == original, search.environ


@test
def duplicate_candidates_and_repeated_setup_preserve_precedence():
    candidates = [r"C:\preferred", r"c:/PREFERRED/.", r"C:\later", "C:\\preferred\\"]
    directories = {
        r"C:\preferred": ["cudart64_13.dll"],
        r"C:\later": ["cudnn64_9.dll"],
    }
    with windows_cuda(directories, candidates, r"C:\Windows") as (module, search):
        module.setup_cuda_dlls()
        expected = r"C:\preferred;C:\later;C:\Windows"
        assert search.environ["PATH"] == expected, search.environ
        module.setup_cuda_dlls()
        assert search.environ["PATH"] == expected, search.environ
        assert search.registered == {r"c:\preferred": 1, r"c:\later": 1}, search.registered


@test
def cuda_twelve_and_thirteen_cufft_wheel_directories_are_searchable():
    directories = {
        r"C:\venv\nvidia\cufft\bin": ["cufft64_11.dll"],
        r"C:\venv\nvidia\cu13\bin\x86_64": ["cufft64_12.dll"],
    }
    with windows_cuda(directories) as (module, search):
        module.setup_cuda_dlls()
        assert search.environ["PATH"].split(";")[:2] == list(directories), search.environ
        assert search.resolve("cufft64_11.dll") == r"c:\venv\nvidia\cufft\bin\cufft64_11.dll"
        assert search.resolve("cufft64_12.dll") == r"c:\venv\nvidia\cu13\bin\x86_64\cufft64_12.dll"


@test
def split_cuda_and_modern_cudnn_dirs_support_secure_dependency_search():
    cuda = r"C:\venv\nvidia\cu13\bin\x86_64"
    cudnn = r"C:\venv\nvidia\cudnn\bin"
    directories = {
        cuda: ["cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"],
        cudnn: ["cudnn64_9.dll", "cudnn_graph64_9.dll", "cudnn_ops64_9.dll"],
    }
    with windows_cuda(directories) as (module, search):
        # A sibling cuDNN component must be reachable even during the FIRST
        # preload. Merely prepending PATH does not affect ctypes' default mode.
        search.probe = "cudnn_graph64_9.dll"
        module.setup_cuda_dlls()
        expected = r"c:\venv\nvidia\cudnn\bin\cudnn_graph64_9.dll"
        assert search.probe_results and all(p == expected for p in search.probe_results), search.probe_results
        assert search.resolve("cudnn_ops64_9.dll") == r"c:\venv\nvidia\cudnn\bin\cudnn_ops64_9.dll"
        assert search.environ["PATH"].split(";")[:2] == [cuda, cudnn], search.environ


@test
def discovery_keeps_explicit_project_then_both_wheel_layouts():
    module = subject()
    with tempfile.TemporaryDirectory(prefix="providers-") as temporary:
        root = Path(temporary)
        project = root / "project"
        prefix = root / "venv"
        site = prefix / "Lib" / "site-packages"
        wheel_dirs = [site / "nvidia" / "cu13" / "bin" / "x86_64",
                      site / "nvidia" / "cudnn" / "bin",
                      site / "nvidia" / "cuda_runtime" / "bin"]
        for directory in wheel_dirs:
            directory.mkdir(parents=True)
        config = ModuleType("app.config")
        config.BASE_DIR = project
        config.settings = SimpleNamespace(cuda_dll_path=str(root / "explicit"))
        isolated_os = SimpleNamespace(path=os.path, sys=SimpleNamespace(prefix=str(prefix)))
        # Exercise real wheel globs, without inspecting other user installations.
        wheel_glob = lambda pattern: glob.glob(pattern) if str(site) in pattern else []
        with patch.dict(sys.modules, {"app.config": config}), \
                patch.object(module, "os", isolated_os), \
                patch.object(module, "glob", SimpleNamespace(glob=wheel_glob)):
            directories = module._candidate_dirs()
        assert directories[:2] == [str(root / "explicit"), str(project / "bin" / "cuda")], directories
        for directory in wheel_dirs:
            assert str(directory) in directories, (directory, directories)


@test
def native_load_failures_leave_cpu_fallback_available():
    runtime = ModuleType("onnxruntime")
    runtime.get_available_providers = lambda: ["CPUExecutionProvider"]
    with windows_cuda({r"C:\incompatible": ["cudnn64_9.dll"]}) as (module, search), \
            patch.dict(sys.modules, {"onnxruntime": runtime}):
        search.fail_loads = True
        module.setup_cuda_dlls()
        assert module.resolve_providers("auto") == ["CPUExecutionProvider"]
        assert module.resolve_providers("cuda") == ["CPUExecutionProvider"]
        runtime.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert module.resolve_providers("cpu") == ["CPUExecutionProvider"]
        assert module.resolve_providers("auto") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


@test
def non_windows_setup_leaves_search_environment_untouched():
    module = subject()
    isolated_os = SimpleNamespace(name="posix", environ={"PATH": "/usr/bin"})
    with patch.object(module, "os", isolated_os):
        module.setup_cuda_dlls()
    assert isolated_os.environ == {"PATH": "/usr/bin"}


@test
def directory_registration_errors_do_not_break_cpu_startup():
    runtime = ModuleType("onnxruntime")
    runtime.get_available_providers = lambda: ["CPUExecutionProvider"]
    with windows_cuda({r"C:\restricted": ["cudart64_13.dll"]}) as (module, search), \
            patch.dict(sys.modules, {"onnxruntime": runtime}):
        def refused_directory(directory):
            raise OSError("directory registration unavailable")

        module.os.add_dll_directory = refused_directory
        module.setup_cuda_dlls()
        assert search.environ["PATH"].split(";")[0] == r"C:\restricted", search.environ
        assert module.resolve_providers("auto") == ["CPUExecutionProvider"]


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"{len(TESTS)-failed}/{len(TESTS)} passed")
    raise SystemExit(bool(failed))
