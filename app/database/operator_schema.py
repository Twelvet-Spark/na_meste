"""Frozen additive operator schemas; attendance's writer owns commit/backup.

Validate each committed predecessor against its own DDL before upgrading.
Recording metadata is installed explicitly by version 3, never via create_all.
"""
from app.operator_settings import legacy_ai_camera_ids
from app.time_utils import utcnow

VERSION = 8
LEGACY_EMPLOYEE_DDL = """CREATE TABLE employees (
    id INTEGER NOT NULL,
    name VARCHAR(150) NOT NULL,
    external_id VARCHAR(64),
    active BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id)
)"""
EMPLOYEE_INDEX = "CREATE INDEX ix_employees_external_id ON employees (external_id)"
EMPLOYEE_ADDITIONS = (
    "first_name VARCHAR(74)",
    "last_name VARCHAR(74)",
    "revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0)",
)
# SQLite inserts added columns before table-level constraints in sqlite_master.
EMPLOYEE_V1_DDL = """CREATE TABLE employees (
    id INTEGER NOT NULL,
    name VARCHAR(150) NOT NULL,
    external_id VARCHAR(64),
    active BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    first_name VARCHAR(74),
    last_name VARCHAR(74),
    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    PRIMARY KEY (id)
)"""
V1_TABLES = {
    "operator_schema": """CREATE TABLE operator_schema (
        id INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL CHECK (version >= 0),
        updated_at DATETIME NOT NULL
    )""",
    "workplaces": """CREATE TABLE workplaces (
        id INTEGER NOT NULL PRIMARY KEY,
        name VARCHAR(150) NOT NULL CHECK (length(name) BETWEEN 1 AND 150 AND name = trim(name)),
        name_key VARCHAR(450) NOT NULL,
        active BOOLEAN NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        UNIQUE (name_key)
    )""",
    "employee_workplaces": """CREATE TABLE employee_workplaces (
        employee_id INTEGER NOT NULL,
        workplace_id INTEGER NOT NULL,
        PRIMARY KEY (employee_id, workplace_id),
        FOREIGN KEY(employee_id) REFERENCES employees (id) ON DELETE CASCADE,
        FOREIGN KEY(workplace_id) REFERENCES workplaces (id) ON DELETE RESTRICT
    )""",
    "operator_requests": """CREATE TABLE operator_requests (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        operation VARCHAR(100) NOT NULL,
        payload_sha256 VARCHAR(64) NOT NULL,
        result JSON NOT NULL,
        created_at DATETIME NOT NULL
    )""",
}
V2_TABLES = {
    **V1_TABLES,
    "camera_archives": """CREATE TABLE camera_archives (
        camera_id INTEGER NOT NULL PRIMARY KEY,
        http_scheme VARCHAR(5) NOT NULL CHECK (http_scheme IN ('http', 'https')),
        http_port INTEGER NOT NULL CHECK (http_port BETWEEN 1 AND 65535),
        utc_offset_minutes INTEGER NOT NULL CHECK (utc_offset_minutes BETWEEN -720 AND 840),
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        updated_at DATETIME NOT NULL,
        FOREIGN KEY(camera_id) REFERENCES cameras (id) ON DELETE RESTRICT
    )""",
}
V3_TABLES = {
    **V2_TABLES,
    "recording_sessions": """CREATE TABLE recording_sessions (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        camera_id INTEGER NOT NULL,
        camera_name VARCHAR(100) NOT NULL,
        source_revision INTEGER NOT NULL,
        requested_at DATETIME NOT NULL,
        started_at DATETIME,
        deadline_at DATETIME NOT NULL,
        ended_at DATETIME,
        status VARCHAR(20) NOT NULL CHECK (status IN ('STARTING','RECORDING','RECONNECTING','STOPPING','STOPPED','COMPLETED','FAILED','INTERRUPTED')),
        reason_code VARCHAR(64),
        FOREIGN KEY(camera_id) REFERENCES cameras (id) ON DELETE RESTRICT
    )""",
    "recording_connections": """CREATE TABLE recording_connections (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        session_id VARCHAR(36) NOT NULL,
        directory VARCHAR(200) NOT NULL UNIQUE,
        source_revision INTEGER NOT NULL,
        created_at DATETIME NOT NULL,
        anchor_utc DATETIME,
        media_origin_seconds FLOAT,
        closed_at DATETIME,
        FOREIGN KEY(session_id) REFERENCES recording_sessions (id) ON DELETE RESTRICT
    )""",
    "recording_segments": """CREATE TABLE recording_segments (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        session_id VARCHAR(36) NOT NULL,
        connection_id VARCHAR(36) NOT NULL,
        camera_id INTEGER NOT NULL,
        source_revision INTEGER NOT NULL,
        storage_key VARCHAR(250) NOT NULL UNIQUE,
        started_at DATETIME,
        ended_at DATETIME,
        duration_ms INTEGER NOT NULL DEFAULT 0 CHECK (duration_ms >= 0),
        bytes INTEGER NOT NULL DEFAULT 0 CHECK (bytes >= 0),
        codec VARCHAR(32),
        state VARCHAR(20) NOT NULL CHECK (state IN ('READY','EXPIRED','MISSING','CORRUPT','DELETING','UNINDEXED')),
        CHECK (state != 'READY' OR (started_at IS NOT NULL AND ended_at IS NOT NULL AND duration_ms > 0)),
        FOREIGN KEY(session_id) REFERENCES recording_sessions (id) ON DELETE RESTRICT,
        FOREIGN KEY(connection_id) REFERENCES recording_connections (id) ON DELETE RESTRICT,
        FOREIGN KEY(camera_id) REFERENCES cameras (id) ON DELETE RESTRICT
    )""",
    "recording_gaps": """CREATE TABLE recording_gaps (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        session_id VARCHAR(36) NOT NULL,
        started_at DATETIME NOT NULL,
        ended_at DATETIME,
        reason_code VARCHAR(64) NOT NULL,
        FOREIGN KEY(session_id) REFERENCES recording_sessions (id) ON DELETE RESTRICT
    )""",
}
V3_INDEXES = {
    "uq_recording_active_camera": """CREATE UNIQUE INDEX uq_recording_active_camera
        ON recording_sessions (camera_id)
        WHERE status IN ('STARTING','RECORDING','RECONNECTING','STOPPING')""",
}
V4_TABLES = {
    **V3_TABLES,
    "face_model_metadata": """CREATE TABLE face_model_metadata (
        face_id INTEGER NOT NULL PRIMARY KEY,
        model_id VARCHAR(64) NOT NULL CHECK (length(model_id) = 64),
        alignment_id VARCHAR(40) NOT NULL CHECK (length(alignment_id) BETWEEN 1 AND 40),
        created_at DATETIME NOT NULL,
        FOREIGN KEY(face_id) REFERENCES employee_faces (id) ON DELETE CASCADE
    )""",
    "camera_scenes": """CREATE TABLE camera_scenes (
        camera_id INTEGER NOT NULL PRIMARY KEY,
        revision INTEGER NOT NULL CHECK (revision >= 0),
        source_revision INTEGER NOT NULL CHECK (source_revision >= 0),
        reference_width INTEGER NOT NULL CHECK (reference_width > 0),
        reference_height INTEGER NOT NULL CHECK (reference_height > 0),
        geometry JSON NOT NULL CHECK (json_valid(geometry)),
        updated_at DATETIME NOT NULL,
        FOREIGN KEY(camera_id) REFERENCES cameras (id) ON DELETE CASCADE
    )""",
    "human_visits": """CREATE TABLE human_visits (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        camera_id INTEGER NOT NULL,
        tracking_generation VARCHAR(36) NOT NULL,
        track_id INTEGER NOT NULL,
        source_revision INTEGER NOT NULL CHECK (source_revision >= 1),
        stream_id VARCHAR(36) NOT NULL,
        started_at DATETIME NOT NULL,
        last_seen_at DATETIME NOT NULL CHECK (last_seen_at >= started_at),
        ended_at DATETIME,
        end_reason VARCHAR(64),
        status VARCHAR(20) NOT NULL CHECK (status IN ('PROVISIONAL','ACTIVE','CLOSED')),
        employee_id INTEGER,
        identity_state VARCHAR(20) NOT NULL CHECK (identity_state IN ('UNKNOWN','FACE_CONFIRMED','CONFLICT')),
        global_id INTEGER,
        last_bbox JSON NOT NULL CHECK (json_valid(last_bbox) AND json_type(last_bbox) = 'array' AND json_array_length(last_bbox) = 4),
        max_person_score FLOAT NOT NULL CHECK (max_person_score BETWEEN 0 AND 1),
        snapshot BLOB CHECK (snapshot IS NULL OR length(snapshot) <= 262144),
        face_thumbnail BLOB CHECK (face_thumbnail IS NULL OR length(face_thumbnail) <= 262144),
        face_quality JSON,
        image_expires_at DATETIME NOT NULL,
        images_purged_at DATETIME,
        CHECK (ended_at IS NULL OR ended_at >= last_seen_at),
        CHECK (status != 'CLOSED' OR ended_at IS NOT NULL),
        UNIQUE (camera_id, tracking_generation, track_id),
        FOREIGN KEY(camera_id) REFERENCES cameras (id) ON DELETE RESTRICT,
        FOREIGN KEY(employee_id) REFERENCES employees (id) ON DELETE RESTRICT,
        FOREIGN KEY(global_id) REFERENCES global_persons (id) ON DELETE SET NULL
    )""",
    "human_visit_events": """CREATE TABLE human_visit_events (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        visit_id VARCHAR(36) NOT NULL,
        dedupe_key VARCHAR(200) NOT NULL UNIQUE CHECK (length(dedupe_key) BETWEEN 1 AND 200),
        type VARCHAR(24) NOT NULL CHECK (type IN ('FIRST_SEEN','IDENTIFIED','IDENTITY_CONFLICT','LOST','RECOVERED','CLOSED','LINKED')),
        occurred_at DATETIME NOT NULL,
        payload JSON NOT NULL CHECK (json_valid(payload) AND json_type(payload) = 'object' AND length(CAST(payload AS BLOB)) <= 16384),
        FOREIGN KEY(visit_id) REFERENCES human_visits (id) ON DELETE CASCADE
    )""",
    "human_journal_gaps": """CREATE TABLE human_journal_gaps (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        camera_id INTEGER NOT NULL,
        tracking_generation VARCHAR(36) NOT NULL,
        started_at DATETIME NOT NULL,
        ended_at DATETIME,
        reason VARCHAR(64) NOT NULL,
        CHECK (ended_at IS NULL OR ended_at >= started_at),
        FOREIGN KEY(camera_id) REFERENCES cameras (id) ON DELETE RESTRICT
    )""",
}
V4_INDEXES = {
    **V3_INDEXES,
    "ix_human_visits_last_seen_id": "CREATE INDEX ix_human_visits_last_seen_id ON human_visits (last_seen_at, id)",
    "ix_human_visits_camera_last_seen": "CREATE INDEX ix_human_visits_camera_last_seen ON human_visits (camera_id, last_seen_at)",
    "ix_human_visits_employee_last_seen": "CREATE INDEX ix_human_visits_employee_last_seen ON human_visits (employee_id, last_seen_at)",
    "ix_human_visit_events_visit_occurred": "CREATE INDEX ix_human_visit_events_visit_occurred ON human_visit_events (visit_id, occurred_at, id)",
    "ix_human_journal_gaps_camera_started": "CREATE INDEX ix_human_journal_gaps_camera_started ON human_journal_gaps (camera_id, started_at)",
}
V5_TABLES = {
    **V4_TABLES,
    "camera_profiles": """CREATE TABLE camera_profiles (
        camera_id INTEGER NOT NULL PRIMARY KEY,
        workplace_id INTEGER,
        zone VARCHAR(8) NOT NULL CHECK (zone IN ('INTERIOR','ENTRANCE')),
        ai_enabled BOOLEAN NOT NULL CHECK (ai_enabled IN (0,1)),
        ai_fps FLOAT CHECK (ai_fps IS NULL OR (ai_fps > 0 AND ai_fps <= 30)),
        revision INTEGER NOT NULL CHECK (revision >= 0),
        updated_at DATETIME NOT NULL,
        FOREIGN KEY(camera_id) REFERENCES cameras (id) ON DELETE CASCADE,
        FOREIGN KEY(workplace_id) REFERENCES workplaces (id) ON DELETE RESTRICT
    )""",
}
V5_INDEXES = {
    **V4_INDEXES,
    "ix_camera_profiles_workplace": "CREATE INDEX ix_camera_profiles_workplace ON camera_profiles (workplace_id)",
}
V6_CAMERA_PROFILES_DDL = """CREATE TABLE camera_profiles (
        camera_id INTEGER NOT NULL PRIMARY KEY,
        workplace_id INTEGER,
        zone VARCHAR(8) NOT NULL CHECK (zone IN ('INTERIOR','ENTRANCE')),
        ai_enabled BOOLEAN NOT NULL CHECK (ai_enabled IN (0,1)),
        ai_fps FLOAT CHECK (ai_fps IS NULL OR (ai_fps > 0 AND ai_fps <= 30)),
        time_tracking BOOLEAN NOT NULL CHECK (time_tracking IN (0,1) AND (time_tracking = 0 OR ai_enabled = 1)),
        revision INTEGER NOT NULL CHECK (revision >= 0),
        updated_at DATETIME NOT NULL,
        FOREIGN KEY(camera_id) REFERENCES cameras (id) ON DELETE CASCADE,
        FOREIGN KEY(workplace_id) REFERENCES workplaces (id) ON DELETE RESTRICT
    )"""
V6_TABLES = {**V5_TABLES, "camera_profiles": V6_CAMERA_PROFILES_DDL}
V6_INDEXES = V5_INDEXES
V7_TABLES = {
    **V6_TABLES,
    "human_visit_identity_revisions": """CREATE TABLE human_visit_identity_revisions (
        id INTEGER NOT NULL PRIMARY KEY,
        visit_id VARCHAR(36) NOT NULL,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        mode VARCHAR(8) NOT NULL CHECK (mode IN ('EMPLOYEE','UNKNOWN','AI')),
        employee_id INTEGER,
        actor VARCHAR(100) NOT NULL,
        reason VARCHAR(1000) NOT NULL,
        created_at DATETIME NOT NULL,
        request_id VARCHAR(36) NOT NULL UNIQUE,
        UNIQUE (visit_id, revision),
        CHECK ((mode = 'EMPLOYEE' AND employee_id IS NOT NULL) OR (mode != 'EMPLOYEE' AND employee_id IS NULL)),
        FOREIGN KEY(visit_id) REFERENCES human_visits (id) ON DELETE RESTRICT,
        FOREIGN KEY(employee_id) REFERENCES employees (id) ON DELETE RESTRICT
    )""",
}
V7_INDEXES = {
    **V6_INDEXES,
    "ix_visit_identity_revision": "CREATE INDEX ix_visit_identity_revision ON human_visit_identity_revisions (visit_id, revision)",
}


def _sha(column):
    return (f"typeof({column}) = 'text' AND length({column}) = 64 AND "
            f"length(CAST({column} AS BLOB)) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'")


def _ms(column):
    """Naive UTC text with at most millisecond precision (SQLAlchemy may pad zeros to six digits)."""
    return (f"typeof({column}) = 'text' AND datetime({column}) IS NOT NULL AND substr({column}, 11, 1) = ' ' "
            f"AND length({column}) IN (19, 23, 26) AND (length({column}) < 26 OR substr({column}, 24, 3) = '000')")


def _rectangle(column):
    kinds = " AND ".join(f"json_type({column}, '$[{i}]') IN ('integer', 'real')" for i in range(4))
    return (f"json_valid({column}) AND json_type({column}) = 'array' AND json_array_length({column}) = 4 "
            f"AND {kinds} AND json_extract({column}, '$[0]') >= 0 AND json_extract({column}, '$[0]') < "
            f"json_extract({column}, '$[2]') AND json_extract({column}, '$[2]') <= 1 "
            f"AND json_extract({column}, '$[1]') >= 0 AND json_extract({column}, '$[1]') < "
            f"json_extract({column}, '$[3]') AND json_extract({column}, '$[3]') <= 1")


def _audit(column, limit):
    return (f"typeof({column}) = 'text' AND length({column}) BETWEEN 1 AND {limit} AND "
            f"instr({column}, char(0)) = 0 AND trim({column}, ' ' || char(9, 10, 11, 12, 13)) != ''")


V8_TABLES = {
    **V7_TABLES,
    "operator_learning_runtime": """CREATE TABLE operator_learning_runtime (
        id INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
        revision INTEGER NOT NULL DEFAULT 0 CHECK (typeof(revision) = 'integer' AND revision >= 0)
    )""",
    "human_visit_photo_metadata": f"""CREATE TABLE human_visit_photo_metadata (
        visit_id VARCHAR(36) NOT NULL PRIMARY KEY,
        sha256 VARCHAR(64) NOT NULL CHECK ({_sha('sha256')}),
        captured_at DATETIME NOT NULL CHECK ({_ms('captured_at')}),
        frame_seq INTEGER NOT NULL CHECK (typeof(frame_seq) = 'integer' AND frame_seq >= 0),
        native_width INTEGER NOT NULL CHECK (typeof(native_width) = 'integer' AND native_width > 0),
        native_height INTEGER NOT NULL CHECK (typeof(native_height) = 'integer' AND native_height > 0),
        FOREIGN KEY(visit_id) REFERENCES human_visits (id) ON DELETE RESTRICT
    )""",
    "recognition_feedback": f"""CREATE TABLE recognition_feedback (
        id VARCHAR(36) NOT NULL PRIMARY KEY CHECK (length(id) = 36),
        visit_id VARCHAR(36) NOT NULL,
        employee_id INTEGER NOT NULL,
        action VARCHAR(6) NOT NULL CHECK (action IN ('ACCEPT', 'REJECT')),
        source VARCHAR(10) NOT NULL CHECK (source IN ('PHOTO_CROP', 'UPLOAD')),
        source_sha256 VARCHAR(64) NOT NULL CHECK ({_sha('source_sha256')}),
        image_sha256 VARCHAR(64) NOT NULL CHECK ({_sha('image_sha256')}),
        crop JSON NOT NULL CHECK ({_rectangle('crop')}),
        captured_at DATETIME CHECK (captured_at IS NULL OR ({_ms('captured_at')})),
        image_jpeg BLOB CHECK (image_jpeg IS NULL OR (typeof(image_jpeg) = 'blob' AND length(image_jpeg) <= 262144)),
        actor VARCHAR(100) NOT NULL CHECK ({_audit('actor', 100)}),
        reason VARCHAR(1000) NOT NULL CHECK ({_audit('reason', 1000)}),
        created_at DATETIME NOT NULL,
        identity_revision INTEGER CHECK (identity_revision IS NULL OR identity_revision >= 1),
        face_id INTEGER CHECK (face_id IS NULL OR face_id >= 1),
        face_key VARCHAR(64) CHECK (face_key IS NULL OR ({_sha('face_key')})),
        face_status VARCHAR(24) NOT NULL CHECK (face_status IN ('ADDED', 'NEGATIVE_ADDED', 'INELIGIBLE', 'LIMIT', 'CONFLICT', 'UNAVAILABLE')),
        revoked_at DATETIME,
        revoked_by VARCHAR(100),
        revoke_reason VARCHAR(1000),
        CHECK ((face_id IS NULL) = (face_key IS NULL)),
        CHECK ((revoked_at IS NULL AND revoked_by IS NULL AND revoke_reason IS NULL) OR (revoked_at IS NOT NULL AND revoked_by IS NOT NULL AND revoke_reason IS NOT NULL AND {_audit('revoked_by', 100)} AND {_audit('revoke_reason', 1000)})),
        FOREIGN KEY(visit_id) REFERENCES human_visits (id) ON DELETE RESTRICT,
        FOREIGN KEY(employee_id) REFERENCES employees (id) ON DELETE RESTRICT
    )""",
    "recognition_examples": f"""CREATE TABLE recognition_examples (
        id INTEGER NOT NULL PRIMARY KEY,
        feedback_id VARCHAR(36) NOT NULL,
        modality VARCHAR(4) NOT NULL CHECK (modality IN ('BODY', 'FACE')),
        polarity VARCHAR(8) NOT NULL CHECK (polarity IN ('POSITIVE', 'NEGATIVE')),
        model_id VARCHAR(64) NOT NULL CHECK ({_sha('model_id')}),
        alignment_id VARCHAR(64) CHECK (alignment_id IS NULL OR length(alignment_id) BETWEEN 1 AND 64),
        embedding BLOB NOT NULL CHECK (typeof(embedding) = 'blob' AND length(embedding) = 2048),
        CHECK ((modality = 'BODY' AND alignment_id IS NULL) OR (modality = 'FACE' AND alignment_id IS NOT NULL)),
        CHECK (polarity = 'NEGATIVE' OR modality = 'BODY'),
        UNIQUE (feedback_id, modality),
        FOREIGN KEY(feedback_id) REFERENCES recognition_feedback (id) ON DELETE RESTRICT
    )""",
}
V8_INDEXES = {
    **V7_INDEXES,
    "ix_recognition_feedback_employee_created": "CREATE INDEX ix_recognition_feedback_employee_created ON recognition_feedback (employee_id, created_at, id)",
    "ix_recognition_feedback_visit_created": "CREATE INDEX ix_recognition_feedback_visit_created ON recognition_feedback (visit_id, created_at, id)",
    "ix_recognition_examples_model": "CREATE INDEX ix_recognition_examples_model ON recognition_examples (modality, model_id, feedback_id)",
}
INDEXES = {1: {}, 2: {}, 3: V3_INDEXES, 4: V4_INDEXES, 5: V5_INDEXES, 6: V6_INDEXES, 7: V7_INDEXES}
INDEXES[8] = V8_INDEXES
SCHEMAS = {1: V1_TABLES, 2: V2_TABLES, 3: V3_TABLES, 4: V4_TABLES, 5: V5_TABLES, 6: V6_TABLES, 7: V7_TABLES,
           8: V8_TABLES}
TABLES = frozenset(V8_TABLES)


def _refuse(detail):
    return RuntimeError(f"Operator schema refused: {detail}. Restore a verified backup or repair explicitly.")


def preflight(connection):
    from app.database.attendance_schema import _catalog, _sql_key

    catalog = _catalog(connection)
    owned = {name for name, (_kind, _sql, table) in catalog.items()
             if table in TABLES or name.startswith("operator_")}
    installed = 0
    if owned:
        if "operator_schema" not in catalog:
            raise _refuse("partial installation without a version")
        try:
            versions = connection.exec_driver_sql("SELECT id, version FROM operator_schema").all()
        except Exception as exc:
            raise _refuse("unreadable version") from exc
        if len(versions) != 1 or versions[0][0] != 1 or versions[0][1] not in SCHEMAS:
            raise _refuse("unknown or uncommitted version")
        installed = versions[0][1]
        expected_tables = SCHEMAS[installed]
        expected_indexes = INDEXES[installed]
        if owned != set(expected_tables) | set(expected_indexes):
            raise _refuse("partial or unexpected operator objects")
        for name, expected in expected_tables.items():
            stored = catalog.get(name)
            if stored is None or stored[0] != "table" or _sql_key(stored[1] or "") != _sql_key(expected):
                raise _refuse(f"table {name} differs from committed version {installed}")
        for name, expected in expected_indexes.items():
            stored = catalog.get(name)
            if stored is None or stored[0] != "index" or _sql_key(stored[1] or "") != _sql_key(expected):
                raise _refuse(f"index {name} differs from committed version {installed}")
    employee = catalog.get("employees")
    if employee is None:
        if installed:
            raise _refuse("employees missing from committed installation")
        return 0
    expected = EMPLOYEE_V1_DDL if installed else LEGACY_EMPLOYEE_DDL
    if employee[0] != "table" or _sql_key(employee[1] or "") != _sql_key(expected):
        raise _refuse("employees differs from recognized legacy/operator version")
    indexes = {name: stored for name, stored in catalog.items() if stored[2] == "employees" and name != "employees"}
    if set(indexes) != {"ix_employees_external_id"}:
        raise _refuse("unexpected or missing employee indexes/triggers")
    if _sql_key(indexes["ix_employees_external_id"][1] or "") != _sql_key(EMPLOYEE_INDEX):
        raise _refuse("employee external ID index differs")
    return installed


def create_legacy_employees(connection):
    connection.exec_driver_sql(LEGACY_EMPLOYEE_DDL)
    connection.exec_driver_sql(EMPLOYEE_INDEX)


def install(connection, installed=0):
    if installed == 0:
        for addition in EMPLOYEE_ADDITIONS:
            connection.exec_driver_sql(f"ALTER TABLE employees ADD COLUMN {addition}")
        for ddl in V1_TABLES.values():
            connection.exec_driver_sql(ddl)
        connection.exec_driver_sql(
            "INSERT INTO operator_schema (id,version,updated_at) VALUES (1,1,?)", (utcnow().isoformat(sep=" "),))
        installed = 1
    if installed == 1:
        connection.exec_driver_sql(V2_TABLES["camera_archives"])
        connection.exec_driver_sql(
            "UPDATE operator_schema SET version=2,updated_at=? WHERE id=1", (utcnow().isoformat(sep=" "),))
        installed = 2
    if installed == 2:
        for name, ddl in V3_TABLES.items():
            if name not in V2_TABLES:
                connection.exec_driver_sql(ddl)
        for ddl in V3_INDEXES.values():
            connection.exec_driver_sql(ddl)
        connection.exec_driver_sql(
            "UPDATE operator_schema SET version=3,updated_at=? WHERE id=1", (utcnow().isoformat(sep=" "),))
        installed = 3
    if installed == 3:
        for name, ddl in V4_TABLES.items():
            if name not in V3_TABLES:
                connection.exec_driver_sql(ddl)
        for name, ddl in V4_INDEXES.items():
            if name not in V3_INDEXES:
                connection.exec_driver_sql(ddl)
        connection.exec_driver_sql(
            "UPDATE operator_schema SET version=4,updated_at=? WHERE id=1", (utcnow().isoformat(sep=" "),))
        installed = 4
    if installed == 4:
        for name, ddl in V5_TABLES.items():
            if name not in V4_TABLES:
                connection.exec_driver_sql(ddl)
        for name, ddl in V5_INDEXES.items():
            if name not in V4_INDEXES:
                connection.exec_driver_sql(ddl)
        now = utcnow().isoformat(sep=" ")
        # AI_CAMERAS is retired: its selection becomes each camera's own AI switch.
        legacy = legacy_ai_camera_ids()
        cameras = connection.exec_driver_sql(
            "SELECT cameras.id, attendance_cameras.role FROM cameras "
            "LEFT JOIN attendance_cameras ON attendance_cameras.camera_id = cameras.id").all()
        for camera_id, role in cameras:
            connection.exec_driver_sql(
                "INSERT INTO camera_profiles (camera_id, workplace_id, zone, ai_enabled, ai_fps, revision, "
                "updated_at) VALUES (?, NULL, ?, ?, NULL, 0, ?)",
                (camera_id, "ENTRANCE" if role is not None else "INTERIOR",
                 int(legacy is None or camera_id in legacy), now))
        # Scene geometry v2 adds doors; existing scenes have none.
        connection.exec_driver_sql(
            "UPDATE camera_scenes SET geometry = json_set(json_set(geometry, '$.version', 2), "
            "'$.doors', json('[]'))")
        connection.exec_driver_sql("UPDATE operator_schema SET version=5,updated_at=? WHERE id=1", (now,))
        installed = 5
    if installed == 5:
        now = utcnow().isoformat(sep=" ")
        connection.exec_driver_sql("DROP INDEX ix_camera_profiles_workplace")
        connection.exec_driver_sql("ALTER TABLE camera_profiles RENAME TO camera_profiles_v5")
        connection.exec_driver_sql(V6_TABLES["camera_profiles"])
        # Cameras that already ran AI start tracking time.
        connection.exec_driver_sql(
            "INSERT INTO camera_profiles (camera_id, workplace_id, zone, ai_enabled, ai_fps, time_tracking, "
            "revision, updated_at) SELECT camera_id, workplace_id, zone, ai_enabled, ai_fps, ai_enabled, "
            "revision, updated_at FROM camera_profiles_v5")
        connection.exec_driver_sql("DROP TABLE camera_profiles_v5")
        connection.exec_driver_sql(V6_INDEXES["ix_camera_profiles_workplace"])
        connection.exec_driver_sql("UPDATE operator_schema SET version=6,updated_at=? WHERE id=1", (now,))
        installed = 6
    if installed == 6:
        connection.exec_driver_sql(V7_TABLES["human_visit_identity_revisions"])
        connection.exec_driver_sql(V7_INDEXES["ix_visit_identity_revision"])
        connection.exec_driver_sql("UPDATE operator_schema SET version=7,updated_at=? WHERE id=1",
                                   (utcnow().isoformat(sep=" "),))
        installed = 7
    if installed == 7:
        now = utcnow().isoformat(sep=" ")
        for name, ddl in V8_TABLES.items():
            if name not in V7_TABLES:
                connection.exec_driver_sql(ddl)
        for name, ddl in V8_INDEXES.items():
            if name not in V7_INDEXES:
                connection.exec_driver_sql(ddl)
        connection.exec_driver_sql("INSERT INTO operator_learning_runtime (id, revision) VALUES (1, 0)")
        connection.exec_driver_sql("UPDATE operator_schema SET version=8,updated_at=? WHERE id=1", (now,))
