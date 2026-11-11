"""Immutable evidence ledger, audited decisions and deterministic work projection.

Every write owns a fresh BEGIN IMMEDIATE connection/savepoint. No Session crosses
threads. Raw evidence never changes; effective decisions and derived conflicts
are separate. Neither absence nor process lifetime is an employee departure.
"""
import hashlib
import json
import logging
from dataclasses import fields, is_dataclass
from datetime import datetime
from itertools import groupby
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from fastapi import HTTPException
from sqlalchemy import delete, func
from sqlalchemy.exc import SQLAlchemyError

from app.attendance_types import EventView, GapView, milliseconds
from app.config import BASE_DIR, settings
from app.database.database import immediate_session
from app.database.models import (
    AttendanceCamera, AttendanceCommission, AttendanceCoverageGap, AttendanceDecision,
    AttendanceEmployeeReference, AttendanceEvent, AttendanceEvidence, AttendanceRequest,
    AttendanceRuntime, Camera, Employee, WorkSession,
)
from app.services.attendance_evidence import evidence_expires_at, purge_evidence
from app.services.attendance_health import AttendanceHealth
from app.services.attendance_queries import event_view, evaluate_session, gap_view, load_trust_blockers
from app.time_utils import attendance_iso, utcnow

logger = logging.getLogger("app.attendance")


def _json(value):
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    if is_dataclass(value):
        return {field.name: _json(getattr(value, field.name)) for field in fields(value)
                if not field.name.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    return value


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _effective(row):
    return {key: _json(getattr(row, key)) for key in (
        "employee_id", "employee_name", "direction", "occurred_at", "status", "revision", "reason_codes")}


class AttendanceService:
    def __init__(self, engine, notifier=None, *, now=utcnow):
        self.engine, self.notifier, self._now = engine, notifier, now
        self.health = AttendanceHealth(now=now)

    def configuration_fingerprint(self, db):
        cameras = []
        for row in db.query(AttendanceCamera).filter_by(enabled=True).order_by(AttendanceCamera.camera_id):
            cameras.append({key: getattr(row, key) for key in (
                "camera_id", "role", "revision", "source_revision", "line_start", "line_end",
                "inside_point", "reference_width", "reference_height")})
        names = ("ai_fps", "frame_max_age", "max_frame_gap", "face_threshold", "face_margin",
                 "face_confirmations", "face_window", "face_interval", "face_min_size",
                 "face_det_score", "face_blur_min")
        thresholds = {name: getattr(settings, f"attendance_{name}") for name in names}
        providers = self.health.snapshot()["providers"]
        face_root = Path(settings.insightface_root) / "models" / settings.insightface_model
        paths = [BASE_DIR / settings.yolo_model]
        face_root = face_root if face_root.is_absolute() else BASE_DIR / face_root
        paths.extend(sorted(face_root.glob("*.onnx")))
        if providers["reid"]:
            paths.append(BASE_DIR / settings.reid_model)
        models = []
        for path in paths:
            try:
                stat = path.stat()
                models.append(dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
            except OSError:
                models.append(dict(path=str(path.resolve()), size=None, mtime_ns=None))
        return dict(version=1, cameras=cameras, thresholds=thresholds, models=models,
                    face_model=str(face_root.resolve()), providers=providers)

    def _commission_valid(self, db):
        runtime = db.get(AttendanceRuntime, 1)
        if runtime.commission_id is None:
            return False
        commission = db.get(AttendanceCommission, runtime.commission_id)
        fingerprint = self.configuration_fingerprint(db)
        providers = fingerprint["providers"]
        cuda = all("CUDAExecutionProvider" in providers[key] for key in (
            "yolo", "live_face_detection", "live_face_recognition",
            "enroll_face_detection", "enroll_face_recognition"))
        roles = {row["role"] for row in fingerprint["cameras"]}
        valid = (commission is not None and cuda and roles == {"ENTRY", "EXIT"}
                 and commission.configuration_fingerprint == fingerprint)
        if not valid:
            runtime.commission_id = None
        return valid

    @staticmethod
    def _clock_blocked(db):
        return any("CLOCK_DISCONTINUITY" in row.reason_codes for row in
                   db.query(AttendanceCoverageGap).filter_by(reviewed=False))

    @staticmethod
    def _reference(db, event_id, employee_id, name=None):
        if employee_id is None or db.get(AttendanceEmployeeReference, (event_id, employee_id)) is not None:
            return
        employee = db.get(Employee, employee_id)
        if employee is not None:
            db.add(AttendanceEmployeeReference(event_id=event_id, employee_id=employee_id,
                                                employee_name=name if name is not None else employee.name))

    def record_crossing(self, crossing):
        camera_id, stream, generation, track = crossing.track_key
        source_key = f"camera:{camera_id}:{stream}:{generation}:{track}:{crossing.crossing_index}"
        identifier = str(uuid5(NAMESPACE_URL, source_key))
        raw = _json(crossing)
        raw.update(version=1, would_auto_accept=crossing.would_auto_accept,
                   candidate_employee_ids=sorted({employee for face in crossing.face_evidence
                                                 for employee in (face.candidate_employee_id, face.runner_up_employee_id,
                                                                  face.confirmed_employee_id) if employee is not None}
                                                | ({crossing.confirmed_employee_id} if crossing.confirmed_employee_id else set())))
        raw = _canonical(raw)
        notification = None
        with immediate_session(self.engine) as db:
            existing = db.query(AttendanceEvent).filter_by(source_key=source_key).one_or_none()
            if existing is not None:
                if existing.raw != raw:
                    raise HTTPException(409, "Source key already has different immutable evidence")
                return existing.id
            runtime = db.get(AttendanceRuntime, 1)
            camera = db.get(Camera, camera_id)
            config = db.get(AttendanceCamera, camera_id)
            if camera is None or config is None:
                raise HTTPException(409, "Captured camera no longer exists")
            employee = db.get(Employee, crossing.confirmed_employee_id) if crossing.confirmed_employee_id else None
            reasons = set(crossing.disqualifying_reasons)
            if not crossing.auto_eligible and not reasons:
                reasons.add("CAPTURE_INELIGIBLE")
            if crossing.direction not in ("ENTRY", "EXIT"):
                reasons.add("INCOMPLETE_CROSSING")
            if employee is None:
                reasons.add("NO_FACE")
            elif not employee.active:
                reasons.add("EMPLOYEE_INACTIVE")
            if runtime.gallery_revision != crossing.gallery_revision:
                reasons.add("GALLERY_CHANGED")
            if config.source_revision != crossing.source_revision:
                reasons.add("SOURCE_CHANGED")
            if config.revision != crossing.calibration_revision:
                reasons.add("CALIBRATION_CHANGED")
            if not config.enabled or not camera.enabled or config.role != crossing.direction:
                reasons.add("CAMERA_UNCONFIGURED")
            if not self._commission_valid(db):
                reasons.add("COMMISSIONING_REQUIRED")
            snapshot = self.health.snapshot()
            if not snapshot["technical_ready"]:
                reasons.update(reason for reason in snapshot["reasons"] if reason != "COMMISSIONING_REQUIRED")
            if self.health.pending_faults():
                reasons.add("COVERAGE_UNRECONCILED")
            if self._clock_blocked(db):
                reasons.add("CLOCK_DISCONTINUITY")
            if db.query(AttendanceCoverageGap).filter_by(ended_at=None).first() is not None:
                reasons.add("COVERAGE_UNAVAILABLE")
            if crossing.before_jpeg is None or crossing.after_jpeg is None:
                reasons.add("EVIDENCE_UNAVAILABLE")
            accepted = not reasons
            previous_latest = db.query(func.max(AttendanceEvent.occurred_at)).filter_by(
                employee_id=employee.id if employee else None, status="ACCEPTED").scalar()
            row = AttendanceEvent(
                id=identifier, source_key=source_key, source="CAMERA", camera_id=camera_id,
                camera_name=camera.name, stream_id=stream, tracking_generation=generation,
                track_id=track, crossing_index=crossing.crossing_index, recorded_at=self._now(), raw=raw,
                employee_id=employee.id if accepted else None,
                employee_name=crossing.confirmed_employee_name if accepted else None,
                direction=crossing.direction, occurred_at=milliseconds(crossing.occurred_at),
                status="ACCEPTED" if accepted else "REVIEW", revision=0,
                reason_codes=sorted(reasons), projection_reasons=[],
            )
            db.add(row)
            db.flush()
            db.add(AttendanceEvidence(event_id=identifier, before_jpeg=crossing.before_jpeg,
                                      after_jpeg=crossing.after_jpeg, face_jpeg=crossing.face_jpeg,
                                      expires_at=evidence_expires_at(row.recorded_at)))
            if runtime.gallery_revision == crossing.gallery_revision:
                names = {}
                for face in crossing.face_evidence:
                    for employee_id, name in ((face.candidate_employee_id, face.candidate_employee_name),
                                              (face.runner_up_employee_id, face.runner_up_employee_name),
                                              (face.confirmed_employee_id, face.confirmed_employee_name)):
                        if employee_id is not None:
                            names[employee_id] = name
                for employee_id in raw["candidate_employee_ids"]:
                    self._reference(db, identifier, employee_id, names.get(employee_id))
            if accepted:
                self._reference(db, identifier, employee.id, row.employee_name)
                self.rebuild_employee(db, employee.id)
                if previous_latest is None or row.occurred_at > previous_latest:
                    session = db.query(WorkSession).filter(
                        (WorkSession.entry_event_id == identifier) | (WorkSession.exit_event_id == identifier)).one_or_none()
                    if session is not None:
                        view = evaluate_session(session, load_trust_blockers(db, as_of=self._now()), as_of=self._now())
                        if view.status != "REVIEW":
                            verb = "Приход" if row.direction == "ENTRY" else "Уход"
                            notification = f"{verb}: {row.employee_name}, {attendance_iso(row.occurred_at)}"
        if notification and self.notifier is not None and settings.telegram_notify_presence:
            try:
                self.notifier.send_text(notification)
            except Exception:
                logger.exception("Attendance notification failed after committed accounting")
        return identifier

    def rebuild_employee(self, db, employee_id):
        rows = db.query(AttendanceEvent).filter_by(employee_id=employee_id).order_by(
            AttendanceEvent.occurred_at, AttendanceEvent.id).all()
        for row in rows:
            row.projection_reasons = []
        db.execute(delete(WorkSession).where(WorkSession.employee_id == employee_id))
        db.flush()
        opened = None
        for _stamp, grouped in groupby((row for row in rows if row.status == "ACCEPTED"), key=lambda row: row.occurred_at):
            group = list(grouped)
            if len(group) > 1:
                for row in group:
                    row.projection_reasons = ["SIMULTANEOUS_EVENTS"]
                if opened is not None:
                    opened.reason_codes = sorted(set(opened.reason_codes) | {"SIMULTANEOUS_EVENTS"})
                    opened.status = "REVIEW"
                continue
            row = group[0]
            if row.direction == "ENTRY":
                if opened is not None:
                    row.projection_reasons = ["DUPLICATE_ENTRY"]
                    opened.reason_codes = sorted(set(opened.reason_codes) | {"DUPLICATE_ENTRY"})
                    opened.status = "REVIEW"
                else:
                    opened = WorkSession(entry_event_id=row.id, employee_id=employee_id,
                                         employee_name=row.employee_name, started_at=row.occurred_at,
                                         ended_at=None, status="OPEN", reason_codes=[])
                    db.add(opened)
            elif row.direction == "EXIT":
                if opened is None:
                    row.projection_reasons = ["MISSING_ENTRY"]
                else:
                    opened.exit_event_id, opened.ended_at = row.id, row.occurred_at
                    opened.status = "REVIEW" if opened.reason_codes else "CLOSED"
                    db.flush()  # release the partial-unique open slot before the next entry
                    opened = None
        db.flush()

    @staticmethod
    def _request(db, operation, payload, target=None):
        data = payload.model_dump(mode="json")
        if target is not None:
            data["target"] = target
        data = _canonical(data)
        identifier = str(payload.request_id)
        old = db.get(AttendanceRequest, identifier)
        if old is not None:
            if old.operation != operation or old.payload != data:
                raise HTTPException(409, "Request UUID already used with different payload or target")
            return old, True
        row = AttendanceRequest(id=identifier, operation=operation, payload=data, result={})
        db.add(row)
        db.flush()
        return row, False

    @staticmethod
    def _employee(db, identifier):
        employee = db.get(Employee, identifier) if identifier is not None else None
        if employee is None:
            raise HTTPException(404, "Employee not found")
        return employee

    @staticmethod
    def _event(db, identifier, revision):
        row = db.get(AttendanceEvent, identifier)
        if row is None:
            raise HTTPException(404, "Event not found")
        if row.revision != revision:
            raise HTTPException(409, "Event changed; refresh before reapplying")
        return row

    def _decision(self, db, request, row, payload, before):
        db.add(AttendanceDecision(request_id=request.id, event_id=row.id, revision=row.revision,
                                   actor=payload.actor, reason=payload.reason, before=before,
                                   after=_effective(row), created_at=self._now()))

    def create_manual_event(self, payload):
        try:
            with immediate_session(self.engine) as db:
                request, replay = self._request(db, "manual", payload)
                if replay:
                    return EventView.model_validate(request.result)
                employee = self._employee(db, payload.employee_id)
                when = milliseconds(payload.occurred_at)
                if when > milliseconds(self._now()):
                    raise HTTPException(422, "Future events are not allowed")
                key = f"manual:{payload.request_id}"
                row = AttendanceEvent(id=str(uuid5(NAMESPACE_URL, key)), source_key=key, source="MANUAL",
                                      raw={"version": 1, "candidate_employee_ids": []}, recorded_at=self._now(),
                                      employee_id=employee.id, employee_name=employee.name,
                                      direction=payload.direction, occurred_at=when, status="ACCEPTED",
                                      revision=0, reason_codes=[], projection_reasons=[])
                db.add(row)
                db.flush()
                self._reference(db, row.id, employee.id, employee.name)
                self._decision(db, request, row, payload, {})
                self.rebuild_employee(db, employee.id)
                result = event_view(db, row)
                request.result = result.model_dump(mode="json")
                return result
        except SQLAlchemyError as exc:
            raise HTTPException(503, "Attendance storage unavailable") from exc

    def decide_event(self, event_id, decision):
        try:
            with immediate_session(self.engine) as db:
                request, replay = self._request(db, "decision", decision, event_id)
                if replay:
                    return EventView.model_validate(request.result)
                row = self._event(db, event_id, decision.expected_revision)
                before = _effective(row)
                previous = row.employee_id
                self._reference(db, row.id, previous, row.employee_name)
                if decision.action == "ACCEPT":
                    employee_id = decision.employee_id if decision.employee_id is not None else row.employee_id
                    if employee_id is None:
                        raise HTTPException(422, "An employee is required")
                    employee = self._employee(db, employee_id)
                    direction = decision.direction or row.direction
                    when = milliseconds(decision.occurred_at) if decision.occurred_at is not None else row.occurred_at
                    if direction not in ("ENTRY", "EXIT") or when > milliseconds(self._now()):
                        raise HTTPException(422, "A direction and non-future timestamp are required")
                    row.employee_id, row.employee_name = employee.id, employee.name
                    row.direction, row.occurred_at, row.status = direction, when, "ACCEPTED"
                    self._reference(db, row.id, employee.id, employee.name)
                else:
                    row.status = decision.action
                    if decision.action == "VISITOR":
                        row.employee_id, row.employee_name = None, None
                row.reason_codes, row.projection_reasons = [], []
                row.revision += 1
                self._decision(db, request, row, decision, before)
                db.flush()
                # Release the old employee's stable entry/unique-exit keys
                # before assigning the same event to the new projection.
                for employee_id in dict.fromkeys((previous, row.employee_id)):
                    if employee_id is not None:
                        self.rebuild_employee(db, employee_id)
                result = event_view(db, row)
                request.result = result.model_dump(mode="json")
                return result
        except SQLAlchemyError as exc:
            raise HTTPException(503, "Attendance storage unavailable") from exc

    def mark_visitors(self, payload):
        try:
            with immediate_session(self.engine) as db:
                request, replay = self._request(db, "visitors", payload)
                if replay:
                    return [EventView.model_validate(item) for item in request.result]
                rows = []
                for item in payload.items:
                    row = self._event(db, item.event_id, item.expected_revision)
                    if row.status != "REVIEW" or row.employee_id is not None:
                        raise HTTPException(409, "Only unassigned review crossings can be bulk visitors")
                    rows.append(row)
                for row in rows:
                    before = _effective(row)
                    row.status, row.revision = "VISITOR", row.revision + 1
                    row.employee_id, row.employee_name = None, None
                    row.reason_codes, row.projection_reasons = [], []
                    self._decision(db, request, row, payload, before)
                db.flush()
                result = [event_view(db, row) for row in rows]
                request.result = [item.model_dump(mode="json") for item in result]
                return result
        except SQLAlchemyError as exc:
            raise HTTPException(503, "Attendance storage unavailable") from exc

    def review_gap(self, gap_id, payload):
        try:
            with immediate_session(self.engine) as db:
                request, replay = self._request(db, "gap", payload, gap_id)
                if replay:
                    return GapView.model_validate(request.result)
                row = db.get(AttendanceCoverageGap, gap_id)
                if row is None:
                    raise HTTPException(404, "Coverage gap not found")
                if row.revision != payload.expected_revision or row.ended_at is None:
                    raise HTTPException(409, "Gap is open or has changed")
                before = gap_view(row).model_dump(mode="json")
                row.reviewed, row.revision = True, row.revision + 1
                result = gap_view(row)
                db.add(AttendanceDecision(request_id=request.id, gap_id=gap_id, revision=row.revision,
                                           actor=payload.actor, reason=payload.reason, before=before,
                                           after=result.model_dump(mode="json"), created_at=self._now()))
                request.result = result.model_dump(mode="json")
                return result
        except SQLAlchemyError as exc:
            raise HTTPException(503, "Attendance storage unavailable") from exc

    @staticmethod
    def _merge_gap(db, start, end, reasons, source_ids=()):
        runtime = db.get(AttendanceRuntime, 1)
        if runtime.monitoring_started_at is None or (end is not None and end < runtime.monitoring_started_at):
            return None
        start = max(start, runtime.monitoring_started_at)
        if end is not None:
            end = max(start, end)
        rows = db.query(AttendanceCoverageGap).filter_by(reviewed=False).order_by(AttendanceCoverageGap.started_at).all()
        overlaps = [row for row in rows if (end is None or row.started_at <= end)
                    and (row.ended_at is None or row.ended_at >= start)]
        if not overlaps:
            row = AttendanceCoverageGap(started_at=start, ended_at=end, reason_codes=sorted(set(reasons)),
                                        source_fault_ids=sorted(set(source_ids)), reviewed=False)
            db.add(row)
            db.flush()
            return row
        row = overlaps[0]
        before = (row.started_at, row.ended_at, row.reason_codes, row.source_fault_ids)
        row.started_at = min(start, *(item.started_at for item in overlaps))
        row.ended_at = None if end is None or any(item.ended_at is None for item in overlaps) else max(
            end, *(item.ended_at for item in overlaps))
        row.reason_codes = sorted(set(reasons).union(*(item.reason_codes for item in overlaps)))
        row.source_fault_ids = sorted(set(source_ids).union(*(item.source_fault_ids for item in overlaps)))
        if before != (row.started_at, row.ended_at, row.reason_codes, row.source_fault_ids):
            row.revision += 1
        for other in overlaps[1:]:
            # Unreviewed rows can still have decisions only after explicit review;
            # they are not deleted once human-reconciled.
            db.delete(other)
        db.flush()
        return row

    def _update_coverage(self, db, healthy, reasons, observed_at, *, end_floor=None):
        runtime = db.get(AttendanceRuntime, 1)
        if runtime.monitoring_started_at is None:
            return
        opened = db.query(AttendanceCoverageGap).filter_by(ended_at=None).one_or_none()
        if not healthy:
            if runtime.heartbeat_at is None or runtime.heartbeat_at < runtime.monitoring_started_at:
                observed_at = min(observed_at, runtime.monitoring_started_at)
            self._merge_gap(db, observed_at, None, reasons or ["COVERAGE_UNAVAILABLE"])
        elif opened is not None:
            opened.ended_at = max(opened.started_at, observed_at, end_floor or observed_at)
            opened.revision += 1

    def update_coverage(self, healthy, reasons, observed_at):
        with immediate_session(self.engine) as db:
            self._update_coverage(db, healthy, reasons, observed_at)

    def persist_health(self, faults, *, healthy, reasons, observed_at, heartbeat=False, stopped=False):
        """Persist sampled coverage and every supplied notification in one commit."""
        with immediate_session(self.engine) as db:
            runtime = db.get(AttendanceRuntime, 1)
            valid = self._commission_valid(db)
            floor = max(observed_at, runtime.heartbeat_at or observed_at)
            known = ({identifier for row in db.query(AttendanceCoverageGap) for identifier in row.source_fault_ids}
                     if faults else set())
            recovery_ends = []
            for fault, identifiers in faults:
                if set(identifiers) <= known:
                    continue
                if fault.ended_at is not None:
                    floor = max(floor, fault.ended_at)
                    recovery_ends.append(fault.ended_at)
                self._merge_gap(db, fault.started_at, fault.ended_at, fault.reason_codes, identifiers)
                known.update(identifiers)
            coverage_reasons = set(reasons)
            if not valid:
                coverage_reasons.add("COMMISSIONING_REQUIRED")
            # Clock review blocks admission, not capture of subsequently healthy
            # coverage: its finite unsafe span must be closable for reconciliation.
            coverage_reasons.discard("CLOCK_DISCONTINUITY")
            coverage_at = observed_at
            opened = db.query(AttendanceCoverageGap).filter_by(ended_at=None).one_or_none()
            if healthy and valid and recovery_ends and opened is not None:
                if "COMMISSIONING_REQUIRED" not in opened.reason_codes:
                    coverage_at = max(recovery_ends)
            # Preserve a backward-clock high-water mark, but do not stretch a
            # recovered sub-tick fault to the monitor's later sampling time.
            close_floor = runtime.heartbeat_at or coverage_at
            self._update_coverage(db, healthy and valid, sorted(coverage_reasons), coverage_at,
                                  end_floor=close_floor)
            if heartbeat or stopped:
                runtime.heartbeat_at = floor
            if stopped:
                runtime.stopped_at = max(observed_at, floor)
            blocked = self._clock_blocked(db)
        return valid, blocked

    def start_monitoring_process(self, now):
        with immediate_session(self.engine) as db:
            runtime = db.get(AttendanceRuntime, 1)
            if runtime.monitoring_started_at is not None:
                start = runtime.stopped_at or runtime.heartbeat_at or runtime.monitoring_started_at
                self._merge_gap(db, min(start, now), max(start, now), ["PROCESS_DOWNTIME"])
            runtime.stopped_at = None
            runtime.heartbeat_at = max(now, runtime.heartbeat_at or now)

    def purge_evidence(self, now):
        return purge_evidence(self.engine, now)
