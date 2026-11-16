# Portable Setup Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Track execution in the native todo tool; checkboxes below describe steps, not a second progress ledger.

**Goal:** Export/import camera credentials, workplaces and enrolled employees/photos between installations, never videos/history, importing only into an empty destination.

**Architecture:** A bounded authenticated encrypted logical document, an atomic SQLite setup-transfer service, and localhost-only Settings forms. Existing database schemas, gallery publication and normal startup remain authoritative; no DB replacement or archive extraction.

**Tech Stack:** Existing FastAPI, SQLAlchemy/SQLite, Pydantic, Pillow, NumPy, vanilla JS/Jinja; new direct cryptography>=46 dependency.

**Spec:** `docs/specs/2026-11-16-portable-setup.md` (normative field/security/limits contract).

## Execution and constraints

- Proposed new branch `feature/portable-setup` in current checkout, preserving existing user edits in README.md, app/ai/providers.py, requirements.txt, run_server_task.bat, start_server.bat, RUNBOOK.md and topology.json. Only isolated new README/dependency hunks may overlap user-modified files. Use eval commit namespace, never git add/commit/stash/reset.
- Three dependent tasks, implemented inline (native). Tasks below are security/concurrency sensitive but receive **no separate task review**; request one whole-change review after integration, subject to harness delegation policy. No fan-out implementation because each task builds on the previous contract.
- No production DB, settings files, real recorder, model installation or video directory access. All tests/services use temporary database/settings/media roots.
- Maintain attendance schema v1 and operator schema v3. Export never includes runtime/request/history/media tables. Import never overwrites populated tables.
- Direct localhost transfer only, bounded request bodies before parsing, fixed-cost password KDF, safe errors, one transfer slot and no input echo. Exact spec limits apply.
- API/background worker tests stay deterministic and isolated. Use existing executable @test script conventions, not a pytest migration. Main agent owns tests and browser verification; no broad unrelated suites.

## Review focus

1. Fresh running destination must count as empty despite startup heartbeat, while any genuine history/request data blocks import. Task 2 covers both.
2. A client disconnect must not release the concurrency slot before offloaded KDF/import work completes; a retry after a committed but lost response is authenticated and idempotent. Task 2 covers both.
3. A file that authenticates but contains forged IDs, malformed thumbnails, extreme nesting or invalid embeddings must fail without partial rows. Task 1 and 2 cover these.
4. Encrypted export is not authorization: remote/proxied/rebinding requests must be rejected before reading their bodies. Task 2 covers these.
5. Preview must not be reusable after changing file/password, and settings polling must not overwrite transfer or settings drafts. Task 3 covers both.

## Task 1: Bounded encrypted format and setup schema

**Review:** per-task class — persisted format, credentials and biometrics; native execution defers review to final whole-change review.

**Files:** Create `app/setup_transfer.py`, `tests/test_setup_transfer_format.py`; modify only new dependency hunk in `requirements.txt`.

**Interfaces:**
- `TransferError(code: str, status_code: int)` with generic safe messages; never embed input values.
- `SetupDocument`: strict typed representation of the exact spec allowlist, including entity IDs/references, model identifier and binary samples. Password fields excluded from repr; binary fields excluded from repr as well.
- `TransferCounts`: cameras, workplaces, employees, faces, assignments integer counts.
- `validate_document(payload: bytes, *, expected_model: str) -> SetupDocument`.
- `encode_transfer(document: SetupDocument, password: str) -> bytes`.
- `decode_transfer(blob: bytes, password: str, *, expected_model: str) -> SetupDocument`.
- Constants `MAX_FILE_BYTES=64*1024*1024`, `MAX_BODY_BYTES=MAX_FILE_BYTES+16*1024`; one format-v1 header and fixed KDF parameters from spec.

- [ ] Write executable tests with locally generated cropped JPEG and normalized 512-float arrays: round-trip Unicode names, exact passwords/bytes/links; inactive tags and legacy names/counts/null thumbnail preserved. Two exports of same document/password differ yet both decode identically.
- [ ] Add failure cases: wrong password, flipped header/ciphertext/tag byte, truncation, unsupported version/model/schema, duplicate JSON keys/IDs, dangling references, unknown fields, invalid host/archive/calibration, NaN/zero/wrong-sized vectors, malformed base64/image, oversized body/document/arrays and depth>16. Assert safe codes and no password in exceptions/repr. Use a test seam only for large limit boundaries, not weaken production constants.
- [ ] Run `.venv/Scripts/python.exe tests/test_setup_transfer_format.py` to establish expected missing-implementation failure, then implement interfaces and direct dependency. Consult retained Context7 /pyca/cryptography Scrypt/AESGCM references; use supported library APIs, not handwritten cryptography.
- [ ] Enforce bounded UTF-8/JSON nesting scan and output assembly before large allocations; validate structured semantics, full file size including header/tag, and no input-controlled KDF costs. Do not silently normalize/drop source records on export.
- [ ] Run the format script once after implementation; expected exit0 with every assertion passing. Commit only owned changes through eval commit namespace.

## Task 2: Snapshot export, atomic empty-only import and guarded APIs

**Review:** per-task class — credentials, concurrency and database writes; native execution defers review to final whole-change review.

**Files:** Create `app/services/setup_transfer_service.py`, `app/api/setup_transfer.py`, `tests/test_setup_transfer.py`; modify `main.py`. Reuse, do not modify, `immediate_session`, existing models, gallery API and operator guard. Extend test support only if essential, maintaining existing defaults.

**Interfaces:**
- `SetupTransferService(engine, *, embedding_model: str)`.
- `status() -> dict`: counts-free readiness and pending restart flag, max_file_bytes.
- `export(password: str) -> bytes`: explicit read snapshot + allowlisted projection + Task 1 encoder.
- `preview(blob: bytes, password: str) -> dict`: Task 1 decoder + counts/warnings/readiness + sha256.
- `import_setup(blob: bytes, password: str, *, request_id: UUID, expected_sha256: str) -> tuple[dict, bool]`: result and replay flag. Transaction commits before return; safe receipt persisted. Separate post-commit gallery publication in HTTP boundary using app state.
- `require_local_transfer(request: Request) -> None`: spec locality/origin/operator guard before body consumption; no secret body reflection.
- APIRouter prefix `/api/setup-transfer`, status/export/preview/import routes and response/status contracts exactly as spec. Register service in lifespan and router in main.py; no startup probes or new schema install.

- [ ] Write isolated SQLite/ASGI tests: source contains sentinel credentials, legacy employees, inactive tags, calibration plus unrelated attendance/global/media fixture metadata. Export/import yields exact included fields/bytes/IDs, fresh revisions, disabled attendance configs and **zero** imported excluded rows. Source state is unchanged. Assert no real video file reads.
- [ ] Test normal destination startup heartbeat is allowed; any camera/workplace/employee/history/recording/request record blocks. Unknown nonempty table fails closed. Preview then competing camera creation blocks commit; two simultaneous imports produce only one commit; injected mid-insert failure leaves destination domain tables empty.
- [ ] Test lost-response same UUID/digest replay after new service instance; replay still requires correct password/authenticated file. Changed payload/operation UUID conflicts; new UUID on nonempty destination refuses. File digest mismatch refuses. Empty exported document remains well-defined and its receipt still blocks a new unrelated import.
- [ ] Test pre-body remote/forwarded/Host/origin/missing-header denial, IPv4/IPv6 localhost success, content-type and streamed-body limits, multipart excess fields/files, cancellation cleanup and in-flight slot lifetime, concurrent busy429, safe errors without raw Pydantic values. Use sentinel credentials and inspect API/log outputs, not source text.
- [ ] Run `.venv/Scripts/python.exe tests/test_setup_transfer.py` to establish expected failure; implement service/routes and startup wiring. Read existing `EmployeeRoute` bounded receive implementation and tests for patterns; do not create a second generic request framework.
- [ ] Export: explicit single snapshot, allowlist only, no SQL supplied by archive. Import: `BEGIN IMMEDIATE`, same-lock replay/emptiness/insertion/gallery revision/receipt; commit then gallery refresh; do not call ReaderManager.start. Failed post-commit refresh returns committed result with restart warning, not rollback error.
- [ ] Exercise real service thread cancellation while holding transfer slot; retain task ownership until thread completes. Multipart closes encrypted spool on every exit; plaintext never written to disk. Serialization/encryption/DB work cannot block ASGI event loop.
- [ ] Run `.venv/Scripts/python.exe tests/test_setup_transfer.py`; expected exit0. Commit owned changes. No DB migrations or changes to existing source publication APIs are expected.

## Task 3: Settings export/import flow and end-to-end proof

**Review:** per-task class — secret-bearing UI; native execution defers review to final whole-change review.

**Files:** Modify `app/templates/settings.html`, isolated README documentation subsection; create `static/js/setup_transfer.js`. Reuse `static/js/ui.js` and existing stylesheet classes. Modify CSS only if actual rendered proof requires a scoped style.

**Interfaces:** Consumes Task 2 endpoints and safe error codes. Separate forms under Settings; no changes to settings.js state or polling contracts. Uses OperatorUI for structured errors/busy/accessibility and custom fetch for binary export with the same operator header. Transfer preview/request state remains memory-only.

- [ ] Read `skill://impeccable` before UI work. Add Russian export password/confirmation and import file/password/preview/confirmation controls, counts and explicit scope/local-only/encryption/restart warnings from spec. No automatic import on file selection.
- [ ] Preserve file and UUID after uncertain import response for replay; invalidate preview after input edits. UI must display committed=true even with restart warning. Clear secrets on success/cancel and revoke download URLs. Never insert names/errors via innerHTML.
- [ ] Update README with localhost-only source export → copy encrypted file → empty destination preview/import → normal restart; required same app/schema/model and VPN routes; stored crops rather than originals; forgotten password unrecoverable; no history/videos/settings transfer. Keep all pre-existing user hunks.
- [ ] Launch two isolated named supervised app services with separate temp database/operator-settings/media paths, AI/Telegram disabled. Seed dummy offline camera and synthetic face samples only. On browser surfaces at1366x900 and390x844: download actual encrypted file, import wrong password, correct preview, import, inspect employee thumbnails/workplaces/camera metadata with passwords still concealed, reject populated destination, retry same request, preserve dirty settings through polling. Verify local-only error with separate ASGI tests rather than opening production LAN access.
- [ ] Restart only the owned destination service; verify imported enabled camera source is restored using a throwaway localhost video-source fixture if checking real live frames. No external source contact. Verify gallery recognition with stored synthetic samples through actual EmbeddingStore and no inference rerun; manual recording remains off, commissioning remains invalid. Observe startup status and clean shutdown.
- [ ] Main agent runs once after integration: `tests/test_setup_transfer_format.py`, `tests/test_setup_transfer.py`, `tests/test_operator_storage.py`, `tests/test_operator_employees.py`, `tests/test_operator_settings.py`, `tests/test_dahua_archive.py`, `tests/test_attendance_recognition.py` using `.venv/Scripts/python.exe`; each must exit0. Add lifecycle suite only if startup/shutdown behavior is changed beyond service registration. Do not repeat unrelated whole-repository suites.
- [ ] Remove only owned temporary data/services/browser tabs, retain concise observed verification evidence. Use verification-before-completion and final review workflow; fix evidence-backed findings with narrow checks. Commit only owned changes via eval commit namespace. Do not merge/push without the user's instruction.

## Design review and approval

Controller self-check: setup-only and empty-only user choices covered; secrets/biometrics/locality boundary, atomicity/replay, legacy photos, concurrency limits and actual UI/restart proof assigned above. A single builder subagent was not dispatched because this harness forbids delegating one already-scoped slice. No implementation or tests run during planning.

Approval must cover design + plan, password encryption/dependency, localhost-only transfer, 64 MiB bound, manual restart/recommissioning, and native execution on feature/portable-setup in this checkout with existing edits preserved. Native means these marked tasks get no individual review before the final review.
