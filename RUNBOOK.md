# Camera monitoring: operational runbook

Local FastAPI app: Dahua RTSP → detection/tracking/employee faces → calibrated doorway events and operator-reviewed work intervals. Optional OSNet cross-camera observations are separate from attendance. Video recording is **manual**, not always-on.

**No application authentication.** Use localhost or a firewall-restricted trusted LAN/VPN only; never expose the app publicly. An operator name or `X-Operator-Request: 1` is not a verified identity.

## 1. Install and run

**Existing Windows installation:** run `start_server.bat`. It selects `.venv`, then `venv`, and preserves the server's exit code; missing both environments exits with 1. Open **http://localhost:8000** unless `APP_PORT` is changed; API reference: `/docs`.

**Fresh Windows setup**, from Command Prompt at the project root:

```bat
winget install --id=Gyan.FFmpeg -e
rem Reopen the terminal after installing FFmpeg.
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
if not exist .env copy .env.example .env
```

Python 3.11+; both FFmpeg and ffprobe are required for video. Tool search: `FFMPEG_PATH` executable plus sibling ffprobe → project `bin/` → PATH.

**Video support is Windows 10/11, one application process.** Windows Job containment is required before FFmpeg/ffprobe children, including live decoding and connection probes as well as recording/playback. There is no non-Windows containment fallback: installing FFmpeg on Linux/macOS alone does not enable these media operations. Containment/storage failure disables the affected media operation, not employee records or manual accounting.

Requirements select GPU ONNX Runtime. Do not keep both CPU and GPU distributions sharing the `onnxruntime` module; export-only PyTorch belongs in `.model-build`, not the server environment:

- **CPU:** uninstall `onnxruntime` and `onnxruntime-gpu`, then install only `onnxruntime`; select `AI_DEVICE=cpu`.
- **NVIDIA:** uninstall the CPU `onnxruntime`, then reinstall the selected GPU distribution with `--no-deps --force-reinstall` if needed. Match CUDA, cuDNN and the driver to the **actual wheel/build source**, not just the ORT version number. CUDA 12/13 and cuDNN major versions are not interchangeable. See [README CUDA setup](README.md#3-cuda-windows-опционально) and the [official compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

The official runtime-dependency installation option is `python -m pip install "onnxruntime-gpu[cuda,cudnn]"`; retain the version/source chosen for the GPU build. Do not infer a CUDA major from the ORT version alone or mix unrelated NVIDIA wheels. This does not install the NVIDIA driver. Check **loaded model providers** in `/debug` and the startup `AI Provider` log; package/provider availability is not proof of CUDA inference. CPU fallback is possible, but a model-load failure may leave live AI or enrollment unavailable.

Models are separate deployment assets, not included in Git or setup transfer:

| Asset | Behavior |
|---|---|
| `models/store-v1/manifest.json` | Authoritative source hashes, provenance, preprocessing and component readiness; selected by `AI_MODEL_MANIFEST`. |
| `person` | Full D-FINE-S COCO detector. Missing/invalid component is a visible analysis outage. |
| `face_detection` / `face_recognition` | YuNet / AuraFace. Unavailable models or gallery do not suppress anonymous person tracking. |
| `body` / `scene` | Independent optional appearance linking / CPU-only outline assistance. |

Launch from the project root, using the selected venv's `python main.py` or the Windows launcher. Run **one server process**, no reload/multiple uvicorn workers: readers, tracks and queues are process-local. `AI_ENABLED=false` disables live inference, not the separate enrollment engine. No runtime path downloads models; old model-selector environment variables no longer select weights.

Startup automatically restores readers for all saved enabled cameras. Connecting is asynchronous; wait for `ONLINE`. Do **not** edit/save every camera after restart. Manual recording is not resumed.

### Preparing the licensed bundle (separate build environment)

```bat
py -m venv .model-build
.model-build\Scripts\python.exe -m pip install -r tools\model_build_requirements.txt
.model-build\Scripts\python.exe tools\prepare_models.py --output models/store-v1 --device cpu
```

Never install export-only PyTorch dependencies into the server environment. The preparer pins official sources and artifact hashes, retains license notices, checks exported graphs and real inference, and publishes the bundle only after validation. Bundle-backed inference does not download weights at startup. Preparation alone does not activate a bundle or certify capacity. Preserve existing model files; upstream license grants do not independently guarantee training-data rights.

### Unattended Windows launch

`run_server_task.bat` can be started at logon. It selects `.venv`, then `venv`, creates `data/` for `data/app_run.log`, reads `Settings.app_port` once at wrapper startup without opening the DB, waits while that exact port is busy, and retries server exits after 5 seconds. Restart the **wrapper**, not just the server, after changing `APP_PORT`.

Use **only one watcher instance**. A port check is neither an atomic single-instance lock nor proof that the listener is this app. Do not launch multiple wrappers or combine a running watcher with a separate foreground server. Stop both watcher and server before backup, restore or maintenance.

## 2. Connect Dahua devices

The **server**, not the browser, must reach the device. Enable RTSP and obtain its credentials/channel settings. The wizard supports a direct Dahua camera or a Dahua recorder, with CGI channel discovery or manual channels; this is not ONVIF or general network discovery.

At `/` (**Камеры**), choose **Добавить устройство**:

1. Enter a bare hostname/IP, separate RTSP port (normally 554), username and password. Do not enter a URL, credentials or port in the host field. IPv6 is accepted with or without brackets. Blank password while editing retains the saved secret.
2. For a recorder, discover channel titles through CGI or enter channels **1–64** manually. Lack of CGI permission does not prevent manual RTSP setup. Choose a name and `main` (subtype 0) or `sub` (subtype 1) per selected channel.
3. Probe the connection, then save and inspect the live preview. The connection test probes metadata; it does not save/start a reader or prove sustained decoding. Saving an enabled camera starts its reader; disabled cameras can be saved for later. Opening preview does not itself start a stopped reader.
4. Selected recorder channels save in one atomic batch: invalid/duplicate sources reject the batch, without a partial set. Saving a camera does **not** start recording or commission attendance.
5. Configure HTTP/HTTPS and port separately for recorder CGI/archive. Archive playback also needs an explicitly confirmed fixed device UTC offset (**−720…840 minutes**); workplace/browser time is not proof of the recorder's clock. There is no automatic DST or clock adjustment.

The supported RTSP path is:

```text
rtsp://USER:ENCODED_PASSWORD@HOST:PORT/cam/realmonitor?channel=1&subtype=1
```

CGI title discovery, live RTSP, archive search and archive playback are separate capabilities. HTTPS certificates are verified; redirects/system proxies are not used for CGI. Real compatibility depends on device/firmware and permissions, not merely a successful protocol fixture or title lookup.

Changing host, RTSP port, credentials, channel or stream invalidates attendance calibration/admission and stops active recording; recheck and recommission. Archive-only HTTP/offset changes do not rewrite attendance/history. Cameras with history cannot be deleted: disable them.

## 3. Employees and workplaces

`/employees`: search names/external IDs, filter workplace/activity, inspect accepted-photo count and remediation flags. **Добавить сотрудника** opens `/employees/new`.

- Supply first name and surname (each 1–74 trimmed characters), optional external ID/workplace labels, and **1–15 photos**. Selecting files only builds a draft. Creation and final photo-set replacement are atomic: one rejected photo saves nothing.
- Accepted files are still JPEG/PNG, ≤15 MiB and ≤20 million pixels each. Exactly one face is required; smaller face-box dimension ≥`max(80, FACE_MIN_SIZE)` px, detector confidence ≥0.80, valid aligned 112×112 face and Laplacian variance ≥20. Bad alignment, blur, low confidence and a confident match to another active employee **reject**, not merely warn.
- The detail page has data/photos/work time/history, explicit photo save/cancel and conflict recovery. Edit metadata/photo sets using the revision from the latest response. On 409, reload the current version, inspect the retained draft, then explicitly reapply it.
- Only embeddings/aligned thumbnails are retained, not original uploads. Old incomplete records can be repaired without changing their ID/history. Employee deletion is blocked when history exists; deactivate instead.
- Gallery startup retains valid matching model/alignment vectors unchanged, including samples without thumbnails. Other samples are re-embedded only from valid stored 112×112 aligned JPEGs. Unconvertible samples remain stored but cannot match. Check usable versus stored counts and remediation; a compatible restart makes no gallery write or commissioning change.
- Photo preparation shares one nonblocking enrollment lease; overlapping requests return `503 ENROLLMENT_BUSY`, not concurrent unbounded CUDA work. Replaying a committed create request retains its original receipt.

Workplaces are optional labels, not schedules, time zones, geofences or attendance admission. Create one beside the selector or manage them at `/settings`; multiple assignments are supported. Archiving preserves assignments/history. An archived label may be retained or removed from an existing employee, not newly assigned; there is no hard-delete workplace action.

## 4. Attendance versus observation archives

**Current accounting** is `/presence` (**Рабочее время**) and `/api/attendance/*`: calibrated ENTRY/EXIT crossings, employee evidence, operator decisions, confirmed intervals and reports. Camera visibility, global IDs, restart, midnight or a missing frame do not create an entry/exit.

Automatic acceptance remains closed until the installation is calibrated and commissioned on site. Admission is bound to source/geometry revisions, model configuration, thresholds and actual providers; changed inputs require renewed acceptance. Current technical readiness and commissioning are separate. CPU fallback does not meet the automatic-admission CUDA requirement. Follow the actual field acceptance procedure in [ATTENDANCE.md](ATTENDANCE.md#ввод-автоматического-учёта-в-эксплуатацию); this runbook makes no claim that an installation has passed it.

Manual events and operator review remain available without automatic admission. Review unknown crossings, conflicting events and coverage gaps; corrections require an operator signature/reason and preserve the audit trail. Open time is provisional. Reports use `ATTENDANCE_UTC_OFFSET_MINUTES` (default **UTC+05:00**), not browser/server local time. A complete day needs completed coverage and resolved blockers.

**Legacy presence** is read-only `GET /api/presence/legacy`, with `trusted_for_hours=false`. Historical recognition updates/`last_seen_at` never proved continuous visibility or a real departure; an unclosed legacy row is not current presence. Old sessions are not imported into work hours. Removed `/api/presence/active` and old presence-list routes are not the current attendance API; `PRESENCE_END_TIMEOUT` does not control worked hours.

**Global observations** at `/global` are a separate read-only probabilistic Re-ID archive, not attendance authority. No manual merge or employee-assignment mutations remain. Restart preserves old history unchanged but starts an empty live gallery and allocates IDs above the DB maximum: a person can receive a new ID. `GLOBAL_GALLERY_TTL` expires in-memory identities, not stored history; there is no age-based automatic global-history cleanup.

Topology is **opt-in**: adapt `topology.json.example` using app camera IDs, set `TOPOLOGY_CONFIG`, restart. Only explicitly listed directed transitions restrict their min/max time; **unlisted edges remain allowed**, and reverse edges are separate. The file's existence does not enable it, and topology does not define attendance roles or hours.

This is not a payroll-readiness guarantee. Schedules, payroll and overtime calculation are not implemented; face recognition is not anti-spoofing/liveness verification.

## 5. Manual recording and archive playback

On a camera page, explicitly start **Локальная запись** for **1–1440 minutes**, after confirming retention. Default/max request duration is 24 hours, not retention. Stop finalizes the last usable partial segment. Recording uses a separate FFmpeg process, copies H.264/HEVC video without audio, and does not run recognition or generate attendance events.

`RECORDING_ROOT` defaults to `data/recordings` (relative paths resolve from the app root). Defaults: 14-day retention, **150 GB** total media budget, **2 GB** free-space reserve. Active files and playback cache count toward the budget; safe old unoccupied files can expire earlier under pressure. If space cannot be freed safely, recording/playback stops rather than exceeding policy. Timestamps approximate server receipt, not camera exposure.

At `/recordings`, choose camera, `local` or `dahua`, and a search interval up to 24 hours. Browser input/display uses browser time; API timestamps require explicit UTC offsets. Remote Dahua times use the separately confirmed device offset. Access/protocol failures are not reported as a successful empty archive.

Select a segment/time for progressive browser HLS playback. Requires FFmpeg/ffprobe and `libx264`; H.264/HEVC are transcoded without audio. hls.js is served locally, with native HLS where supported. Only **one converter** is allowed; one playback covers at most one hour of continuous recording. ±10-second/time jumps beyond the buffer create a new request; actual gaps require choosing the next recording. Playback speed is 1×.

Cache is capped at 2 GB within the media budget. After cache expiry, resume creates a request at the saved historical time, not live. Leaving/changing playback closes that playback, not manual recording. Interrupted recordings are marked interrupted on restart and are not resumed. Historical local files remain associated with their old source/time; playback never updates work hours or offers employee/face search.

Windows storage ownership and Job containment fail closed. Unsupported/unavailable containment/storage can return 503; this is not a reason to bypass isolation or run extra workers.

## 6. Settings, diagnostics and Telegram

Settings precedence is **process environment > operator JSON > root `.env` > defaults**. They are loaded at startup, not hot-reloaded.

`/settings` edits only allowed next-start values and shows active/pending values. Save does not interrupt cameras/recording or restart the app; restart normally afterwards. Enabled camera readers restore, manual recording does not. Threshold changes require recommissioning without rewriting history. Environment-controlled fields are locked; reset removes only the selected operator override.

The default file is `data/operator-settings.json`; `OPERATOR_SETTINGS_PATH` can place it elsewhere. Version conflicts return 409 and retain the draft for explicit reconciliation. Invalid/unknown operator JSON settings fail startup safely; unknown `.env` keys are ignored. Telegram token fields are write-only: blank keeps the secret, explicit confirmed clearing removes it. Protect the file with OS permissions.

`/debug` refreshes every 5 seconds while visible, marks disconnected data stale, and isolates section failures. Actual diagnostics include tool availability, loaded model providers, attendance readiness/reasons/role FPS/frame age/p95/queues, camera frame health/reconnects, recording/storage usage, media processes/playbacks and the latest CGI/archive result. A green probe does not commission attendance.

Read-only endpoints:

```http
GET /api/debug/status
GET /api/debug/events?after=0&limit=100
GET /api/debug/export
GET /api/attendance/status
GET /api/cameras/1/status
GET /api/recordings/status
```

The support export contains allowlisted diagnostic/settings projections, not a DB backup or raw log/file viewer. The in-memory ring retains up to 200 current-run coded events, without original exceptions, secrets, stream URLs or process arguments.

**Telegram:** configure a BotFather bot, add it to the intended group and preferably set an explicit `TELEGRAM_CHAT_ID`. Otherwise `/start` discovery caches a chat in `data/telegram_chat_id.txt`, which can also be a private chat. Blank token or `TELEGRAM_ENABLED=false` disables notifications; settings changes need restart. `TELEGRAM_NOTIFY_PRESENCE` sends text for confirmed attendance events; optional global-transition notifications can include photos. HTTPS to `api.telegram.org`, a 200-item in-memory queue and three attempts do not guarantee delivery. Images leave this machine.

## 7. API safety and troubleshooting

Send **`X-Operator-Request: 1` on every mutating request**, including camera probes/discovery, recordings/playbacks and settings; any `Origin` must match the app. This is a trusted-network request guard, not authentication. Example using an actual saved camera ID:

```http
POST /api/cameras/1/test
X-Operator-Request: 1
```

Use current bodies/revisions/UUID request IDs from `/docs` and [README examples](README.md#curl-примеры). Do not use obsolete employee JSON creation, individual photo upload/delete or global-person merge examples. Setup transfer has stricter localhost-only rules below.

| Symptom | Check |
|---|---|
| Camera test fails | Server→device access, credentials, ports, channel/stream, tool and Windows containment readiness. |
| Test passes, preview stalls | Camera status frame age, errors/reconnect count; test does not prove sustained decoding. Try a suitable substream and inspect source/network health. |
| Video but unavailable AI/photos | Actual model/provider errors, complete compatible models, `AI_ENABLED`/`AI_CAMERAS`, accepted active employee photos and live face quality. |
| Automatic attendance blocked | `/api/attendance/status`: current reasons, role calibration, commissioning, real CUDA providers, performance/coverage/clock/write faults. Do not lower identity thresholds blindly. |
| Recording/playback unavailable | `/debug`: containment, storage ownership, quota/free space, FFmpeg/ffprobe/libx264, archive permissions/confirmed device offset. |
| Config edit appears ignored | Compare active/pending values and precedence; restart. Do not reconnect every camera manually. |

## 8. Backup and history protection

Stop **watcher and all application processes** before a filesystem backup/restore. A lone copy of a live SQLite `.db` is unsafe: committed changes may still be in `-wal`. Use a consistent SQLite backup procedure or, after stopping, retain the database and any remaining `-wal`/`-shm` as one set. Never delete WAL manually.

Back up the actual `DATABASE_URL` location, `data/`, `.env`, the operator-settings file (including an external override path), models and the topology in use. For a restore including video, also copy the complete `RECORDING_ROOT`, even if outside `data/`; a DB/index cannot recreate missing media. Schema-migration snapshots are not a full installation backup. Verify restoration on a separate installation.

SQLite contains camera credentials, face embeddings/thumbnails, attendance evidence/history, legacy/global observations and recording metadata. Raw video/cache live under the media root. Protect all of them and backups with OS access controls. Employee/camera history prevents deletion (409): deactivate/disable instead. Workplace archiving preserves assignments. Media retention and attendance JPEG-evidence retention are separate from legacy/global history, which is not automatically purged by age.

## 9. Encrypted setup transfer — local only, no video

Use `/settings` → **Перенос на другой компьютер** directly on the app machine through `localhost` (remote desktop is fine). LAN/public/proxy transfer requests are rejected; forwarded headers are not accepted. Export/import still requires the operator header and matching Origin when present.

Use the same application/dependency version (`cryptography>=46` is retained) and compatible declared `face_model_id` on both machines; model files themselves are not exported. V1 is a data migration decoder, not a legacy model backend. V2 checks each sample's actual provenance; compatible vectors remain unchanged, valid legacy thumbnails are re-embedded, and unusable samples remain stored without matching provenance. Preview reports reencoded/reenrollment/scene-review counts.

1. Export a `.nmsetup` with a unique **12–1024 UTF-8-byte** passphrase, repeated in the form. Maximum file size: **64 MiB**. The passphrase cannot be recovered.
2. Move the encrypted file and communicate its password separately.
3. Start the destination with an **empty database**. Preview the file, compare counts/warnings, then explicitly confirm import. Non-empty destinations are rejected: no merge/overwrite.
4. If the response is lost, retry confirmation using the unchanged form/file/password; do not clear it and invent a different request.
5. Restart normally. Enabled camera readers restore, recording does not. Recheck calibration and recommission attendance: automatic admission is not transferred.

Included: camera credentials/live/archive setup, disabled attendance calibration geometry, workplaces/assignments, employee records, saved face thumbnails/embeddings and paired model/alignment provenance, plus descriptive scene geometry/reference dimensions. Imported scenes have source revision zero and require explicit review/resave; visits, history, video and frozen/assistance images remain excluded.

Excluded: video, recording metadata/cache, attendance/observation history and evidence, app/operator settings, `.env`/Telegram, topology, models, Cloudflare/WARP/VPN and routing. Reconfigure network access separately. **This is setup migration, not a full backup.**

## 10. Private offline evaluation

Use `tools/evaluate_cctv.py`, not the server entry point, for local clip evaluation. Its manifest lists local clip/annotation/enrollment paths; it must not contain camera credentials or recorder URLs. Relative paths resolve against the manifest. Keep clips, anonymous labels, thumbnails and reports under ignored `data/evaluation/`, never in Git.

The runner establishes `NA_MESTE_EVALUATION_ROOT` before importing application modules. This internal bootstrap requires an existing absolute temporary directory and bypasses deployment JSON, `.env`, ambient application settings and secret sources. SQLite/media/topology are redirected there and Telegram is disabled. Do not set this variable for the production server.

Decode throughput, analyzed throughput and preview throughput are different measurements. Replicated copies of one view measure load, not multi-store accuracy. Without operator-confirmed enrollment-to-video labels, employee recognition is `NOT_EVALUATED`. Record the actual GPU; an RTX 4070 result cannot certify an RTX 5060 Ti.
