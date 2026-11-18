"""Выбор execution-провайдеров ONNX Runtime и резолвинг CUDA DLL (Windows).

CUDA используется автоматически, если доступна; при любых проблемах —
CPU-фолбэк (приложение не падает).
"""
import ctypes
import glob
import logging
import os

logger = logging.getLogger("app.ai.providers")

# DLL, от которых зависит onnxruntime_providers_cuda.dll (Windows)
_CUDA_DLLS = (
    "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll", "cudnn64_8.dll",
    "cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll", "cufft64_12.dll", "cudnn64_9.dll",
)

# Регистрации должны жить до завершения процесса; ключи — нормализованные
# абсолютные пути, чтобы повторный вызов не регистрировал каталоги заново.
_DLL_DIRECTORY_HANDLES: dict[str, object] = {}


def _candidate_dirs() -> list[str]:
    """Где ищем CUDA/cuDNN DLL: .env → bin/cuda проекта → venv (pip-колёса
    nvidia-*) → torch из соседнего окружения (например, ComfyUI) → тулкит."""
    from app.config import BASE_DIR, settings

    dirs: list[str] = []
    if settings.cuda_dll_path:
        dirs.append(settings.cuda_dll_path)
    dirs.append(str(BASE_DIR / "bin" / "cuda"))

    # pip-колёса nvidia-*-cu12/cu13 внутри venv.
    # CUDA 13 кладёт DLL в nvidia/cu13/bin/x86_64, CUDA 12 — в nvidia/<pkg>/bin.
    site = os.path.join(os.sys.prefix, "Lib", "site-packages")
    for pattern in ("nvidia/*/bin", "nvidia/*/bin/*", "nvidia/**/bin"):
        dirs.extend(glob.glob(os.path.join(site, pattern)))

    # чужие окружения с torch (ComfyUI и т.п.) — там лежат нужные DLL
    dirs.extend(glob.glob(
        os.path.expanduser(
            "~/AppData/Local/Comfy-Desktop/ComfyUI-Installs/*/ComfyUI/.venv/Lib/site-packages/torch/lib"
        )
    ))

    dirs.extend(glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/bin"))
    dirs.extend(glob.glob("C:/Program Files/NVIDIA/CUDNN/*/bin"))
    return dirs


def setup_cuda_dlls() -> None:
    """Подключает директории с CUDA-библиотеками ДО создания ORT-сессий.

    PATH нужен загрузчику ORT, а os.add_dll_directory — безопасному поиску
    зависимостей ctypes (он не использует PATH). Сначала подключаются все
    каталоги, затем DLL предзагружаются. Вызывать до первого InferenceSession.
    """
    if os.name != "nt":
        return
    found: list[tuple[str, str, list[str]]] = []
    seen: set[str] = set()
    for d in _candidate_dirs():
        directory_key = os.path.normcase(os.path.abspath(d))
        if directory_key in seen:
            continue
        seen.add(directory_key)
        if not os.path.isdir(d):
            continue
        dlls = [dll for dll in _CUDA_DLLS if os.path.isfile(os.path.join(d, dll))]
        if dlls:
            found.append((d, directory_key, dlls))
    if not found:
        logger.info("CUDA DLL не найдены — при недоступности CUDA будет использован CPU")
        return
    # Сначала все каталоги в PATH (зависимости DLL могут лежать в соседнем
    # каталоге, например cuBLAS в cu13/bin/x86_64, cuDNN в nvidia/cudnn/bin),
    # затем предзагрузка — она проверяет, что библиотеки действительно грузятся.
    current_path = os.environ.get("PATH", "")
    path_keys = {
        os.path.normcase(os.path.abspath(d))
        for d in current_path.split(os.pathsep) if d
    }
    path_dirs = [d for d, directory_key, _ in found if directory_key not in path_keys]
    if path_dirs:
        os.environ["PATH"] = os.pathsep.join(path_dirs) + os.pathsep + current_path
    # ctypes использует зарегистрированные каталоги, но не PATH. Подключаем
    # соседние зависимости до первой предзагрузки и сохраняем дескрипторы.
    for d, directory_key, _ in found:
        if directory_key not in _DLL_DIRECTORY_HANDLES:
            try:
                _DLL_DIRECTORY_HANDLES[directory_key] = os.add_dll_directory(directory_key)
            except OSError:
                pass  # PATH остаётся доступен ORT; ошибка не должна мешать CPU
    for d, _, dlls in found:
        logger.info("CUDA DLL: %s (%d библиотек)", d, len(dlls))
        for dll in dlls:
            try:
                ctypes.WinDLL(os.path.join(d, dll))
            except OSError:
                pass  # битая/несовместимая — ORT сам сообщит при загрузке EP


def resolve_providers(device: str = "auto") -> list[str]:
    """CUDA при наличии, иначе CPU. Порядок = приоритет."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in available
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda" and not has_cuda:
        logger.warning("CUDA запрошена (AI_DEVICE=cuda), но недоступна — используется CPU")
        return ["CPUExecutionProvider"]
    if has_cuda:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def model_providers(providers: list[str]) -> list:
    """Keep FP32 math truthful; CUDA's default TF32 can exceed model parity."""
    return [(provider, {"use_tf32": 0}) if provider == "CUDAExecutionProvider" else provider
            for provider in providers]


def provider_label(providers: list[str]) -> str:
    return providers[0] if providers else "?"


# Resident model sessions reported by loaded_providers() and shown in diagnostics.
PROVIDER_KEYS = (
    "person_detection", "live_face_detection", "live_face_recognition",
    "enroll_face_detection", "enroll_face_recognition",
    "reid", "scene_encoder", "scene_decoder", "scene_detector",
)
