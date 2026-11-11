"""Frozen additive operator schema v1; attendance's writer owns commit/backup.

Future versions must retain this definition and validate their predecessor before
adding an explicit migration. No media/archive tables belong to version 1.
"""
from app.time_utils import utcnow

VERSION = 1
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
TABLES = frozenset(V1_TABLES)
FUTURE_TABLES = frozenset({"camera_archives", "recording_sessions", "recording_segments",
                           "recording_connections", "recording_gaps"})


def _refuse(detail):
    return RuntimeError(f"Operator schema refused: {detail}. Restore a verified backup or repair explicitly.")


def preflight(connection):
    from app.database.attendance_schema import _catalog, _sql_key

    catalog = _catalog(connection)
    owned = {name for name, (_kind, _sql, table) in catalog.items()
             if table in TABLES | FUTURE_TABLES or name.startswith("operator_")}
    installed = bool(owned)
    if installed:
        if "operator_schema" not in catalog:
            raise _refuse("partial installation without a version")
        try:
            versions = connection.exec_driver_sql("SELECT id, version FROM operator_schema").all()
        except Exception as exc:
            raise _refuse("unreadable version") from exc
        if versions != [(1, VERSION)]:
            raise _refuse("unknown or uncommitted version")
        if owned != TABLES:
            raise _refuse("partial or unexpected operator objects")
        for name, expected in V1_TABLES.items():
            stored = catalog.get(name)
            if stored is None or stored[0] != "table" or _sql_key(stored[1] or "") != _sql_key(expected):
                raise _refuse(f"table {name} differs from committed version 1")
    employee = catalog.get("employees")
    if employee is None:
        if installed:
            raise _refuse("employees missing from committed installation")
        return False
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


def install(connection):
    for addition in EMPLOYEE_ADDITIONS:
        connection.exec_driver_sql(f"ALTER TABLE employees ADD COLUMN {addition}")
    for ddl in V1_TABLES.values():
        connection.exec_driver_sql(ddl)
    connection.exec_driver_sql("INSERT INTO operator_schema (id,version,updated_at) VALUES (1,1,?)", (utcnow(),))
