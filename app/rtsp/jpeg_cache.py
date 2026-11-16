"""Two latest JPEG derivatives per camera, never native images or frame history."""
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Literal

import cv2

from app.rtsp.reader import FramePacket

DerivativeSize = Literal["thumbnail", "preview"]
JPEG_QUALITY = 70
_LONG_EDGE = {"thumbnail": 480, "preview": 1280}


@dataclass(slots=True)
class _CameraJPEGs:
    lock: threading.Lock = field(default_factory=threading.Lock)
    images: dict[str, tuple[tuple[int, str, int, str], bytes]] = field(default_factory=dict)
    pending: tuple[tuple[int, str, int, str], Future[bytes]] | None = None


class JPEGCache:
    """Serialize one camera's encodes; unrelated cameras do not share that lock.

    Invalidating removes the entire camera slot. An in-flight old reader may
    finish its encode, but cannot repopulate the replacement reader's cache.
    Callers must revalidate the reader/stream before publishing its result.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cameras: dict[int, _CameraJPEGs] = {}

    def get(self, camera_id: int, packet: FramePacket, size: DerivativeSize) -> bytes:
        if size not in _LONG_EDGE:
            raise ValueError("Only thumbnail and preview derivatives may be cached")
        with self._lock:
            camera = self._cameras.get(camera_id)
            if camera is None:
                camera = _CameraJPEGs()
                self._cameras[camera_id] = camera
        key = (camera_id, packet.stream_id, packet.seq, size)
        while True:
            with camera.lock:
                previous = camera.images.get(size)
                if previous is not None and previous[0] == key:
                    return previous[1]
                pending = camera.pending
                if pending is None:
                    future: Future[bytes] = Future()
                    camera.pending = (key, future)
                    break
            # Share even an obsolete in-flight key without retaining its image
            # after the requesting callers finish. Only one encode per camera
            # can run, and only two completed derivatives remain resident.
            if pending[0] == key:
                return pending[1].result()
            pending[1].exception()
        try:
            frame = packet.frame
            height, width = frame.shape[:2]
            edge = _LONG_EDGE[size]
            if max(height, width) > edge:
                scale = edge / max(height, width)
                frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                                   interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                raise RuntimeError("JPEG_ENCODING_FAILED")
            jpeg = encoded.tobytes()
            # A delayed request must not evict a newer packet from this stream.
            with camera.lock:
                if previous is None or previous[0][1] != packet.stream_id or previous[0][2] <= packet.seq:
                    camera.images[size] = (key, jpeg)
                future.set_result(jpeg)
                camera.pending = None
        except BaseException as exc:
            with camera.lock:
                future.set_exception(exc)
                camera.pending = None
            raise
        return jpeg

    def invalidate(self, camera_id: int) -> None:
        with self._lock:
            self._cameras.pop(camera_id, None)

    def clear(self) -> None:
        with self._lock:
            self._cameras.clear()
