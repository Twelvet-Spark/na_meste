# Excel Reports and Camera Samples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. User explicitly requested “Create plan then implement it”; no additional approval pause.

**Goal:** Export a workplace-filtered Excel timesheet of counted hours; display truthful person/identity overlays on every preview; add a quality-checked employee face from a frozen native camera frame.
**Architecture:** Keep `compute_day` as the only hours calculation. Extend the existing report snapshot/filter contract and serialize it with openpyxl. Share overlay rendering between directory/detail. Cropping is client-side over the existing native snapshot endpoint and submits to the existing atomic employee-face-set API; no new identity override, snapshot cache, database schema, or camera-area assignment.
**Tech Stack:** FastAPI, SQLAlchemy, vanilla JS/SVG/canvas, openpyxl >=3.1.5,<4.
**Spec:** User requests and the Scope below are authoritative; no separate spec document.
**Execution:** Native/inline on branch `excel-reports-camera-samples` in this checkout; one fresh whole-branch review. No migration/auth changes; enrollment keeps the existing validation, locking and optimistic revision controls.

## Scope and calls made
- Workplace selects employees assigned to it, not a subset of their cameras. Sum all work time already counted for each selected employee. Filters combine date range (1–366 inclusive days), employee, workplace and employee status ALL/ACTIVE/INACTIVE.
- XLSX is a matrix: Фамилия | Имя | Сотрудник | Рабочие места | one column per date | Итого. Keep the full-name column for legacy records whose split names are incomplete; never guess names. Include selected employees with zero observations. Past/today empty days are numeric zero, future days blank. Durations are numeric Excel intervals `[h]:mm`, totals may exceed 24h. Second sheet records calculation parameters/as-of and provisional-today warning. No formulas from user input.
- Existing JSON/CSV daily layout remains available and uses the same filters. Existing counted hours are not changed. UI adds workplace/status filters and separate “Скачать Excel” and CSV links.
- Show person detector score separately from face-match similarity. Appearance identity has a source label, not an invented confidence. Sticky/appearance identities display the employee name even when local recognition state is unknown. Grid previews poll visible online cameras only, clear stale boxes, stop when hidden.
- Crop workflow is a quality-checked face sample, not an instant manual override of a track and not a permanent desk identity. Freeze a native JPEG, select a rectangle by pointer or normalized coordinate controls, choose an employee, preview the crop, explicitly confirm identity, preserve every existing sample, and submit through PUT /api/employees/{id}/faces. Exactly-one-face, minimum face size/quality, duplicate-person protection, 15-sample limit and revision conflict rules remain authoritative. Poor/absent face is rejected, never replaced by a fake embedding. No historical hours are relabeled.

## Global constraints
- Reuse OperatorUI helpers, current Russian design/tokens and existing APIs; no new frontend dependency.
- New library dependency is only openpyxl. Source: https://openpyxl.readthedocs.io/en/stable/datetime.html; durations use numeric day fractions/timedelta and `[h]:mm`. Force untrusted text cells to string type rather than formula/error type.
- Never modify deployment DB/media or connect to real cameras for smoke tests. Use temporary DBs and a synthetic reader/model double, and report that limit.
- File edits use anchored edit; selective commits use eval commit namespace. Leave the three pre-existing untracked files untouched.

## Review focus
1. Employee in multiple workplaces must not duplicate rows/hours; selecting a workplace must retain hours seen on other cameras (Task 1 test).
2. Zero rows, mixed LIVE/FINAL/future dates, >24h totals, exact milliseconds and formula-like names must survive a workbook round trip (Task 1 test).
3. REID/sticky identities must not display as unknown, and stale/offline thumbnails must not keep old boxes (Task 2 checks).
4. Reversed/out-of-image or tiny crop and multiple/no faces must fail safely; native image coordinates must remain correct on a resized display (Task 3 checks).
5. Employee switch/revision conflict or interrupted save must not remove existing photos or silently resubmit stale state (Task 3 checks).

### Task 1: Filtered counted-hours timesheet
**Files:** requirements.txt; app/services/attendance_reports.py; app/api/attendance.py; app/templates/presence.html; static/js/presence.js; tests/test_attendance_api.py; new tests/test_timesheet_export.py.
**Interfaces:** `build_report(db, from_date, to_date, employee_id, as_of, *, workplace_id=None, employee_status="ALL") -> dict` adds `employees` metadata and selected filter values. Existing rows/totals retain their semantics. `report_xlsx(report: dict) -> bytes` consumes that materialized result. GET /api/attendance/report.xlsx accepts the same filters as JSON/CSV, returns application/vnd.openxmlformats-officedocument.spreadsheetml.sheet and attachment `work-time-<from>-<to>.xlsx`.
- [ ] Write regression cases in test_timesheet_export.py: assigned workplace selects its employees but keeps their other-camera hours; status AND employee filter; two workplace memberships do not duplicate hours; selected zero-hour employee present; XLSX roundtrip has real duration 30h, daily 6h, future blank, and formula-like employee string remains text.
- [ ] Run `.venv/Scripts/python.exe tests/test_timesheet_export.py`; observe missing-export/filters failure.
- [ ] Install declared openpyxl dependency; implement report selection/metadata and XLSX serializer without duplicating compute_day. Bind all three export endpoints to identical filters.
- [ ] Add workplace/status controls and Excel link to report UI; preserve selected filters in URL and downloads.
- [ ] Run the new script plus tests/test_attendance_api.py and node --check static/js/presence.js; all pass. Commit own paths.

### Task 2: Truthful shared preview overlays
**Files:** app/ai/worker.py (DetectionState only); static/js/camera.js; static/js/app.js; app/templates/camera.html; app/templates/index.html; new static/js/camera_overlay.js; static/css/style.css (only actual existing style path if different); new tests/js/test_camera_overlay.mjs; relevant detection API tests.
**Interfaces:** additive track fields `person_score: number`, `face_similarity: number|null`; preserve existing confidence for compatibility. `window.CameraOverlay.label(track)` and `render(svg, tracks)` are shared by directory/detail. Known employee_id/name wins over local state. Missing scores render no numeric claim.
- [ ] Add JS regression proving {state:'unknown',employee_id:1,employee_name:'Анна',identity_source:'REID'} labels Анна and appearance, never unknown; confidence 0.91 detector vs 0.82 face remain distinct; unknown has detector score only. Observe RED before helper exists.
- [ ] Publish finite detector and applicable face score in DetectionState; build shared SVG renderer with safe textContent and normalized valid boxes.
- [ ] Reuse renderer in camera detail and each directory image wrapper. Poll only visible online tiles, clear on stale result/offline/hidden page, avoid leaked per-tile intervals during rerender.
- [ ] Run JS test, changed JS syntax, and detection/worker covering scripts; all pass. Commit own paths.

### Task 3: Frozen-frame employee face crop
**Files:** new static/js/camera_sample.js; app/templates/camera.html; existing stylesheet for scoped crop layout; new tests/js/test_camera_sample.mjs; tests/test_operator_employees.py only if an existing consumer contract is uncovered.
**Consumes:** existing GET /api/attendance/cameras/{id}/snapshot (native JPEG + dimensions); GET /api/employees and /{id}; PUT /api/employees/{id}/faces multipart fields `metadata={expected_revision,keep_face_ids}` plus `files`; X-Operator-Request:1. No new endpoint.
- [ ] Add deterministic geometry tests: reverse drag normalizes, coordinates clamp to image, resized pointer maps to native crop, zero-area is rejected. Observe RED.
- [ ] Add explicit section “Фото сотрудника из кадра”, freeze/cancel controls, native image rectangle overlay and keyboard numeric bounds, crop preview, employee selector, identity-confirm checkbox and one save button. Freeze does not stop live reader.
- [ ] Prepare PNG from selected native rectangle; fetch selected employee revision/face IDs, preserve all existing IDs, disable at 15. No save without crop + selected employee + explicit confirmation. Keep draft on error, require reloading employee after 409, never retry writes automatically or claim instant/historical re-identification. Release native image/crop resources on success/cancel/page exit.
- [ ] Explain poor-face rejection and model unavailable states. Render error detail from existing photo validation. Disable controls while loading/saving.
- [ ] Run geometry test, node --check, and existing employee enrollment contracts. Browser proves crop/native coordinates, preservation of old photos, rejection and successful sample append with a deterministic face-engine double. Commit own paths.

### Task 4: Integration verification and delivery
**Files:** README.md, ATTENDANCE.md; throwaway smoke scripts only in this plan's ignored workspace.
- [ ] Update report/export and camera-sample docs, including appearance confidence limits and no automatic desk identity.
- [ ] Run a real app/API/browser against a temporary DB and synthetic reader: filters/zero employees, downloaded workbook roundtrip and >24h total; directory/detail overlays; freeze-drag/coordinate edits, selection confirmation, sample append preserving originals, conflict/model/quality errors, mobile layout.
- [ ] Run every tests/test_*.py and tests/js/*.mjs once after integration. Preserve genuine model-dependent skips.
- [ ] Request one fresh sdd-final-reviewer against the branch and this plan; fix evidence-backed Important/Critical findings with regressions and rerun affected/full checks as needed.
- [ ] Selectively commit changes, remove this plan's throwaway workspace after recording decisions, leave branch unmerged/unpushed. Report actual verification and limits.
