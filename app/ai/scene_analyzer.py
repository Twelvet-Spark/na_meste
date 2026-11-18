"""One-click scene analysis: detector boxes and prompted masks become one reviewable draft.

Nothing here saves anything. The draft is validated as a complete version-2 scene so the
editor can load it as-is; items the geometry rules reject are dropped, never repaired.
"""
from uuid import uuid4

from app.ai.scene_segmenter import SceneSegmentationError
from app.scene_geometry import GeometryError, validate_door_quad, validate_geometry, validate_polygon

MAX_OBJECTS = 24
MAX_DOORS = 8
MAX_NEGATIVE_POINTS = 16
# Rows near the bottom of the frame are the likeliest floor; columns span it.
FLOOR_CANDIDATES = [[x, y] for y in (.72, .84, .95) for x in (.2, .35, .5, .65, .8)]


def _rounded(points):
    # Six decimals is sub-pixel for any camera and bounds the whole document well under its size limit.
    return [[round(x, 6), round(y, 6)] for x, y in points]


def _polygon(points):
    try:
        return validate_polygon(_rounded(points), 'Outline')
    except GeometryError:
        return None


def _rectangle(box):
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _floor(snapshot, segmenter, detections):
    positive = [point for point in FLOOR_CANDIDATES if not any(_inside(point, item.box) for item in detections)]
    if not positive:
        return None
    negative = [[(item.box[0] + item.box[2]) / 2, (item.box[1] + item.box[3]) / 2]
                for item in [item for item in detections if item.kind in ('OBJECT', 'PERSON')][:MAX_NEGATIVE_POINTS]]
    suggestion = segmenter.floor(snapshot, positive, negative)
    return None if suggestion is None else _polygon(suggestion.points)


def _object(snapshot, segmenter, detection):
    for suggestion in segmenter.suggest(snapshot, [], [], list(detection.box)):
        polygon = _polygon(suggestion.points)
        if polygon is not None:
            return dict(polygon=polygon, outline='MASK')
    polygon = _polygon(_rectangle(detection.box))
    return None if polygon is None else dict(polygon=polygon, outline='BOX')


def _door(detection):
    x1, y1, x2, y2 = detection.box
    try:
        return validate_door_quad(_rounded([[x1, y2], [x2, y2], [x2, y1], [x1, y1]]), 'Door')
    except GeometryError:
        return None


def analyze(snapshot, detector, segmenter):
    """Return {floor, objects, doors, warnings}; the caller owns locking and snapshot validity."""
    detections = sorted(detector.detect(snapshot.packet.frame), key=lambda item: item.score, reverse=True)
    floor = _floor(snapshot, segmenter, detections)
    objects = []
    for detection in [item for item in detections if item.kind == 'OBJECT'][:MAX_OBJECTS]:
        outline = _object(snapshot, segmenter, detection)
        if outline is not None:
            objects.append(dict(id=str(uuid4()), label=detection.label, score=detection.score, **outline))
    doors = []
    for detection in [item for item in detections if item.kind in ('DOOR', 'PASSAGE')][:MAX_DOORS]:
        quad = _door(detection)
        if quad is not None:
            doors.append(dict(id=str(uuid4()), label=detection.label, kind=detection.kind,
                              quad=quad, score=detection.score))
    try:
        validate_geometry(dict(
            version=2, floor=floor, metric=None,
            objects=[dict(id=item['id'], label=item['label'], polygon=item['polygon']) for item in objects],
            doors=[dict(id=item['id'], label=item['label'], kind=item['kind'], quad=item['quad'])
                   for item in doors]))
    except GeometryError as exc:
        raise SceneSegmentationError('SCENE_INFERENCE_FAILED', 'Analysis produced an invalid scene') from exc
    warnings = [code for code, missing in (('FLOOR_NOT_FOUND', floor is None), ('NO_DOORS_FOUND', not doors),
                                           ('NO_OBJECTS_FOUND', not objects)) if missing]
    return dict(floor=floor, objects=objects, doors=doors, warnings=warnings)
