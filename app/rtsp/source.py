"""Immutable, transaction-captured camera configuration for reader publication."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.database.models import Camera

CONNECTION_FIELDS = frozenset({"nvr_host", "rtsp_port", "username", "password", "channel", "stream_type"})


@dataclass(frozen=True, slots=True)
class CameraSource:
    id: int
    name: str
    nvr_host: str
    rtsp_port: int
    username: str
    password: str = field(repr=False)
    channel: int
    stream_type: str
    enabled: bool
    created_at: datetime
    updated_at: datetime
    source_revision: int
    # Zero denotes boot restoration; live writes receive a process-local order
    # while holding SQLite's write lock, including name/enabled-only changes.
    publication_sequence: int = 0

    @property
    def has_password(self) -> bool:
        return bool(self.password)

    @classmethod
    def from_camera(cls, camera: "Camera", *, source_revision: int,
                    publication_sequence: int = 0) -> "CameraSource":
        """Call while the camera and its revision share one database snapshot."""
        return cls(id=camera.id, name=camera.name, nvr_host=camera.nvr_host,
                   rtsp_port=camera.rtsp_port, username=camera.username, password=camera.password,
                   channel=camera.channel, stream_type=camera.stream_type, enabled=camera.enabled,
                   created_at=camera.created_at, updated_at=camera.updated_at,
                   source_revision=source_revision, publication_sequence=publication_sequence)
