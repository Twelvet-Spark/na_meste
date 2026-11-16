# Portable setup export/import

## Intent and decisions

Move setup to another PC running the same app without re-entering recorder passwords or enrolling employees again. User selected: **setup only**, and **import only into an empty installation**. Never include video. This is a draft for approval, not implemented behavior.

Proposed interface: Settings → «Перенос на другой компьютер».

```text
Экспорт: [Пароль] [Повтор пароля] [Скачать файл переноса]
Импорт: [Выбрать файл] [Пароль] [Проверить]
        Counts + exclusions + destination readiness → [Импортировать]
        Success → «Перезапустите приложение обычным способом»
```

## Scope

Include explicit allowlisted records:
- Cameras: names, hosts, RTSP ports, usernames/passwords, channels, stream types, enabled flags, creation/update times.
- Archive connection settings: HTTP/HTTPS, port and confirmed fixed device UTC offset; no footage or archive search results.
- Camera calibration geometry/role/reference resolution, if present, for review on the destination. Import attendance calibration **disabled**; never transfer commissioning approval or source/runtime revisions.
- Workplaces: names, active flags, employee assignments, including archived workplaces.
- Employees: canonical/structured names, external IDs, active flags and dates; saved face thumbnails and 512-float recognition embeddings.
- Preserve IDs and relationships because the destination is empty. Initialize fresh local revisions; retain legacy incomplete names, missing thumbnails and legacy sample counts with existing remediation flags, without silently dropping data.

Exclude attendance/presence/work-session history, tracked global people/observations/events, evidence images, all recording metadata, video files, playback caches, request logs, diagnostic logs, commission/runtime state, .env, operator-settings.json, topology files, models, Cloudflare tokens/configuration and machine-specific paths. The app does not retain original enrollment photographs: export only stored face crops and embeddings. This is a setup transfer, **not a complete disaster-recovery backup**.

## Approach and alternatives

Use a versioned logical setup document encrypted into one `.nmsetup` file. Do not copy/replace SQLite files: raw DB copies would carry excluded history/media metadata and conflict with running connections. Do not implement merge: user chose empty-only. No ZIP extraction, pickle, executable SQL, arbitrary filenames or import-time network requests.

Calls proposed by the assistant:
- Password encryption mandatory; same password needed on destination, no recovery mechanism. Add direct `cryptography>=46` dependency.
- Transfer operations available **only through direct localhost access** on the app PC; operators may Remote Desktop to it and open localhost. The existing app lacks authentication, and encrypted downloads alone would not prevent an unauthenticated remote caller exporting with their own password.
- Imported readers start on the next normal app restart, not during import; manual recordings never start automatically. Imported attendance configuration requires explicit verification/recommissioning.
- 64 MiB maximum encrypted transfer file; reject over-limit exports/imports explicitly. This bounds memory and is visible before upload; no silent truncation.

## File format and validation

`magic(8) || version(1) || salt(16) || nonce(12) || ciphertext_and_tag`:
- Magic bytes `b'NMSETUP\x00'`, format version 1.
- Fresh random salt and nonce for every export. Scrypt(length=32, n=32768, r=8, p=1) derives AES-256-GCM key from UTF-8 password; fixed parameters are not controlled by the input file. Header is AES-GCM associated data, 16-byte authentication tag included.
- Password 12–1024 UTF-8 bytes, used exactly as entered (no trimming/normalization); confirmation on export. Never put passwords in URLs, filenames, logs, browser storage, request receipts or plaintext temporary files. Memory cannot be promised securely erased in Python/JS.
- Plaintext: compact UTF-8 JSON with transfer_version, exported_at (explicit UTC), attendance_schema_version=1, operator_schema_version=3, embedding_model (configured InsightFace pack), embedding_encoding='float32-le-512', and the allowlisted entity arrays. Binary fields strict base64. No compression.
- Strict typed fields, reject extra keys, duplicate JSON keys, nonfinite numbers, malformed base64, duplicate/out-of-range IDs, dangling foreign keys and invalid camera/archive/calibration fields. Reject unsupported format/schema/model identifiers rather than guessing. Model-pack name matching is not a guarantee for locally replaced model weights; require the same unmodified model pack on both PCs in documentation.
- Embeddings: exactly 2048 bytes, finite nonzero float32 vector, preserved without rerunning inference; thumbnails null or bounded JPEG/PNG (<=1 MiB and <=1 million pixels), header verification plus decode, never fetched from external paths. Preserve source values unless field representation explicitly requires conversion; invalid stored source data must fail export with a safe error rather than produce an unusable file.
- Limit to 10,000 cameras, 10,000 workplaces, 10,000 employees, 150,000 face rows, 100,000 assignments, nesting depth 16; enforce file/document limits before unbounded materialization and counts before ORM writes. Current 1–15 enrollment limits do not retroactively destroy legacy data.
- IDs are positive integers no greater than 9,007,199,254,740,991, so browser links remain lossless. Embeddings must also pass the existing gallery normalizer without float32 overflow.
- Before JSON decoding, enforce each collection limit, depth 16, at most 32 members per object and a 2,000,000-container/delimiter budget; do not first materialize an oversized object graph.
- Generic wrong-password/damaged-file error; distinguish unsupported format and deployment/model incompatibility using safe codes only. No exception text or Pydantic input echoes.

## Transaction and runtime behavior

Export reads all included records in **one explicit SQLite read transaction**, detached from HTTP/session lifetimes. Query allowlisted columns only; do not touch video roots or include whole ORM objects. Release DB snapshot before encryption. Export does not block recording on purpose or mutate source data.

Preview decrypts/validates and returns safe counts/warnings, encrypted-file SHA256 and destination readiness; no credentials, embeddings or photo bodies. Preview writes nothing and reserves nothing. Commit reuploads the file and password plus request UUID and expected encrypted-file digest; decrypt/revalidate again. Do not store uploaded files or passwords in server-side preview caches.

Import performs replay check, complete emptiness recheck and all inserts inside existing `immediate_session` (BEGIN IMMEDIATE). A preview becoming stale cannot overwrite newly created data. All domain tables must be empty, including employee/camera/history/media/request tables; only known schema markers and the normal AttendanceRuntime singleton are allowed. Runtime startup/heartbeat timestamps are not user data. Do not reset runtime fields. Recognized initialized schema required; never weaken existing schema preflight. Unknown populated tables fail closed.

Use OperatorRequest operation `setup.import`, the commit UUID, encrypted-file SHA256 and only a sanitized result receipt. Same UUID/digest authenticated retry returns original success, including after restart; changed UUID on a populated destination refuses, reused UUID with different payload/operation conflicts. Reauthenticate the file before replay. The receipt is the sole newly introduced request record; no schema migration is needed.

Insert cameras and archives with fresh revision values; insert disabled AttendanceCamera geometry without commissions. Employees start revision 1; increment destination gallery_revision once, then publish committed gallery through EmbeddingStore.refresh. Import works with AI unavailable (no model inference/download required). No ReaderManager.start, recording launch, source probes, attendance events or background job is invoked by importing. A later user action may explicitly start a camera; ordinary startup restores imported enabled cameras. Refresh failure after commit must not claim rollback: return committed success with restart warning. In-process pending-restart status is set after commit; a new process starts with it clear.

A write/constraint failure rolls the entire import back. No destructive backup/replacement is needed because an import cannot alter a non-empty destination. A rollback does not affect runtime health bookkeeping by concurrent normal workers.

## HTTP and security boundary

New `/api/setup-transfer` routes use the existing same-origin/operator-header guard, plus a transfer-specific direct-loopback guard: client IPv4/IPv6 loopback, Host hostname strictly localhost/127.0.0.1/::1 with legal port, reject Forwarded and all X-Forwarded-* / X-Real-IP headers. Apply before parsing request bodies, including status. Never use forwarding headers to grant locality. Clearly reject reverse-proxy and remote access with `TRANSFER_LOCAL_ONLY`; this is not new general app authentication. Trusted local processes/admins remain trusted.

- `GET /status`: available, destination_empty, restart_required, max_file_bytes.
- `POST /export`: JSON `{password}` <=8 KiB; binary download with generic timestamp filename, application/octet-stream, no-store, nosniff.
- `POST /preview`: multipart exactly one file field `file` plus `password`; bounded metadata, file <=64 MiB; `{sha256, counts, warnings, destination_empty}`.
- `POST /import`: same file/password plus request_id and expected_sha256; success 201, authenticated replay 200, `{committed:true, counts, restart_required, warnings}`.
- Safe failures: local-only403, malformed/wrong password/incompatible422, nonempty or request conflict409, oversized413, busy429, DB/encryption unavailable503. Content-Type validation415. Do not reflect any input values.

Bound ASGI receive before multipart spooling to 64 MiB +16 KiB and each scalar field to its explicit limit. Multipart owns/cleans any temporary **encrypted** upload on all exits. All decrypted plaintext stays in bounded memory. One nonblocking per-app transfer operation slot covers body processing through completion; release on cancellation/error, including offloaded work completion (do not release while a cancelled thread is still running). KDF/JSON/DB work runs off the event loop; no unbounded wait queue.

## UI and delivery

Reuse Russian UI, OperatorUI helpers and existing CSS; add a separate setup_transfer.js module, not a framework. Independent export/import forms do not join dirty settings submissions. Password confirmation, explicit secret-data warning and download responsibility; preview counts and exclusions before confirming import. Reject wrong/oversized file early but keep server checks authoritative. Change of selected file/password invalidates preview/request UUID; retry after uncertain network completion preserves UUID/file until resolved. Clear sensitive inputs on success/cancel, revoke download URLs, no localStorage. Network/validation failure retains recoverable form state and never displays a fabricated success. Local-only denial explains opening localhost on the server. Restart is manual; no HTTP restart button.

## Acceptance and limits

Use isolated source/destination SQLite databases, dummy credentials and synthetic enrolled samples; no production DB, real recorder or existing video file access. Round trip must preserve passwords, IDs, inactive workplace links, Unicode names and exact sample bytes while every excluded table remains empty. Test races, replay, damage, limits, compatibility and locality. Actual Settings browser flow must download from source, preview/import on empty destination, show saved employees/thumbnails/cameras, refuse populated destination, preserve dirty settings, and show post-restart readiness. Use only a throwaway local source if checking reader startup; never probe transferred real hostnames.

Existing user changes in README.md, app/ai/providers.py, requirements.txt, launch BAT files, RUNBOOK.md and topology.json remain untouched except isolated new dependency/documentation hunks in README/requirements. No production code changed during design.
