"""Windows process-lifetime containment, established before EVERY media child.

The job contains this application itself. NEVER close its handle on shutdown:
closing the sole handle would kill the application, not merely its children.
"""
import ctypes
import os
import threading
from ctypes import wintypes

from fastapi import HTTPException


class MediaProcessGuard:
    _lock = threading.Lock()
    _initialized = False
    _handle = None

    @classmethod
    def _initialize_once(cls) -> bool:
        with cls._lock:
            if cls._initialized:
                return cls._handle is not None
            cls._initialized = True
            if os.name != "nt":
                return False  # No orphan-prone non-Windows media fallback.
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            size_t = ctypes.c_size_t

            class BasicLimits(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                            ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", size_t), ("MaximumWorkingSetSize", size_t),
                            ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", size_t),
                            ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

            class IOCounters(ctypes.Structure):
                _fields_ = [(name, ctypes.c_ulonglong) for name in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class ExtendedLimits(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
                            ("ProcessMemoryLimit", size_t), ("JobMemoryLimit", size_t),
                            ("PeakProcessMemoryUsed", size_t), ("PeakJobMemoryUsed", size_t)]

            kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            kernel.CreateJobObjectW.restype = wintypes.HANDLE
            kernel.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
            kernel.SetHandleInformation.restype = wintypes.BOOL
            kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            kernel.SetInformationJobObject.restype = wintypes.BOOL
            kernel.GetCurrentProcess.argtypes = []
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            kernel.AssignProcessToJobObject.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            handle = kernel.CreateJobObjectW(None, None)  # unnamed; NULL security => non-inheritable
            if not handle:
                return False
            limits = ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE, NO breakaway flags
            if (not kernel.SetHandleInformation(handle, 1, 0)
                    or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))):
                kernel.CloseHandle(handle)  # Still empty: safe ONLY before successful self assignment.
                return False
            if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
                kernel.CloseHandle(handle)
                return False
            cls._handle = handle  # Retain sole handle until OS process death; no destructor/atexit.
            return True

    @classmethod
    def initialize(cls) -> bool:
        try:
            return cls._initialize_once()
        except (OSError, AttributeError, ValueError):
            # Missing/incompatible Win32 APIs disable media, not the operator UI.
            return False

    @classmethod
    def require(cls) -> None:
        if not cls.initialize():
            raise HTTPException(503, "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE")
