"""Immutable, transaction-captured camera configuration for reader publication."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.database.models import Camera

CONNECTION_FIELDS = frozenset({"nvr_host", "rtsp_port", "username", "password", "channel", "stream_type"})


@dataclass(frozen=True, slots=True)
class ArchiveSettings:
    http_scheme: str
    http_port: int
    utc_offset_minutes: int
    revision: int
    updated_at: datetime

    @classmethod
    def from_archive(cls, archive):
        if archive is None:
            return None
        return cls(archive.http_scheme, archive.http_port, archive.utc_offset_minutes,
                   archive.revision, archive.updated_at)


@dataclass(frozen=True, slots=True)
class ArchiveSource:
    camera_id: int
    source_revision: int
    archive_revision: int
    host: str
    rtsp_port: int
    http_scheme: str
    http_port: int
    username: str = field(repr=False)
    password: str = field(repr=False)
    channel: int
    utc_offset_minutes: int


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
    archive: ArchiveSettings | None = None

    @property
    def has_password(self) -> bool:
        return bool(self.password)

    @classmethod
    def from_camera(cls, camera: "Camera", *, source_revision: int,
                    publication_sequence: int = 0, archive=None) -> "CameraSource":
        """Call while the camera and its revision share one database snapshot."""
        return cls(id=camera.id, name=camera.name, nvr_host=camera.nvr_host,
                   rtsp_port=camera.rtsp_port, username=camera.username, password=camera.password,
                   channel=camera.channel, stream_type=camera.stream_type, enabled=camera.enabled,
                   created_at=camera.created_at, updated_at=camera.updated_at,
                   source_revision=source_revision, publication_sequence=publication_sequence,
                   archive=ArchiveSettings.from_archive(archive))
