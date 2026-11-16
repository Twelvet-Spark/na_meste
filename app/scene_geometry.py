"""Strict scene-v1 JSON and planar geometry in normalized native coordinates.

Objects describe silhouettes; they are not exclusion masks. Measured rectangles
assume a stationary camera, a coplanar floor and no lens-distortion correction.
"""
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from uuid import UUID

import cv2
import numpy as np

MAX_DOCUMENT_BYTES = 128 * 1024
MAX_VERTICES = 128
MAX_OBJECTS = 32
MAX_RECTIFIED_EDGE = 1280
MAX_RECTIFIED_PIXELS = 1_700_000


class GeometryError(ValueError):
    """The complete scene must be rejected without publishing partial geometry."""


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise GeometryError(f'{name} requires exactly: {", ".join(keys)}')


def _number(value, name, *, minimum=0, maximum=1, exclusive_minimum=False):
    if type(value) not in (int, float):
        raise GeometryError(f'{name} must be a finite number, not a boolean')
    try:
        finite = math.isfinite(value)
    except (OverflowError, ValueError):
        finite = False
    if (not finite or value > maximum or value < minimum
            or (exclusive_minimum and value == minimum)):
        raise GeometryError(f'{name} is outside the permitted range')
    return value


def _point(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise GeometryError(f'{name} must be [x,y]')
    return [_number(value[0], name), _number(value[1], name)]


def _orientation(a, b, c):
    """Robust sign, with exact binary-float arithmetic only near cancellation."""
    left = (b[0] - a[0]) * (c[1] - a[1])
    right = (b[1] - a[1]) * (c[0] - a[0])
    determinant = left - right
    if abs(determinant) > 3.3306690738754716e-16 * (abs(left) + abs(right)):
        return 1 if determinant > 0 else -1
    ax, ay = map(Fraction, a)
    bx, by = map(Fraction, b)
    cx, cy = map(Fraction, c)
    exact = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return (exact > 0) - (exact < 0)


def _on_segment(point, a, b):
    return (_orientation(a, b, point) == 0
            and min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= point[1] <= max(a[1], b[1]))


def _intersects(a, b, c, d):
    abc, abd = _orientation(a, b, c), _orientation(a, b, d)
    cda, cdb = _orientation(c, d, a), _orientation(c, d, b)
    return ((abc * abd < 0 and cda * cdb < 0)
            or (abc == 0 and _on_segment(c, a, b))
            or (abd == 0 and _on_segment(d, a, b))
            or (cda == 0 and _on_segment(a, c, d))
            or (cdb == 0 and _on_segment(b, c, d)))


def polygon_area(points):
    """Unsigned normalized image area, without coordinate quantization."""
    x, y = points[0]
    return abs(math.fsum((a[0] - x) * (b[1] - y) - (b[0] - x) * (a[1] - y)
                         for a, b in zip(points, points[1:] + points[:1]))) / 2


def _polygon(value, name):
    if not isinstance(value, list) or not 3 <= len(value) <= MAX_VERTICES:
        raise GeometryError(f'{name} needs 3..{MAX_VERTICES} vertices')
    points = [_point(point, name) for point in value]
    if len({tuple(point) for point in points}) != len(points):
        raise GeometryError(f'{name} contains duplicate vertices')
    count = len(points)
    for i, point in enumerate(points):
        previous, following = points[i - 1], points[(i + 1) % count]
        if _orientation(previous, point, following) == 0 and not _on_segment(point, previous, following):
            raise GeometryError(f'{name} has overlapping adjacent edges')
        for j in range(i + 2, count):
            if i == 0 and j == count - 1:
                continue
            if _intersects(point, following, points[j], points[(j + 1) % count]):
                raise GeometryError(f'{name} self-intersects')
    if polygon_area(points) == 0 and not any(
            _orientation(points[0], points[i], points[i + 1]) for i in range(1, count - 1)):
        raise GeometryError(f'{name} has zero area')
    return points


def point_in_polygon(point, polygon):
    """Boundary-inclusive containment for a simple, possibly concave polygon."""
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if _on_segment(point, a, b):
            return True
        if (a[1] > point[1]) != (b[1] > point[1]):
            if (_orientation(a, b, point) > 0) == (b[1] > a[1]):
                inside = not inside
    return inside


def validate_geometry(value):
    """Return independent JSON-compatible v1 data; never round coordinates.

    This is the single validator for scene saves and encrypted-transfer import.
    Unknown/missing keys at every level are errors, not ignored extensions.
    """
    _keys(value, ('version', 'floor', 'objects', 'metric'), 'Scene')
    if type(value['version']) is not int or value['version'] != 1:
        raise GeometryError('Scene version must be integer 1')
    if not isinstance(value['objects'], list) or len(value['objects']) > MAX_OBJECTS:
        raise GeometryError(f'Scene permits at most {MAX_OBJECTS} objects')
    # Reject non-JSON values and oversized documents before geometric work.
    try:
        size = len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                              separators=(',', ':')).encode('utf-8'))
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise GeometryError('Scene must be finite UTF-8 JSON') from exc
    if size > MAX_DOCUMENT_BYTES:
        raise GeometryError('Scene exceeds 128 KiB')
    floor = None if value['floor'] is None else _polygon(value['floor'], 'Floor')
    objects, identifiers = [], set()
    for item in value['objects']:
        _keys(item, ('id', 'label', 'polygon'), 'Object')
        if not isinstance(item['id'], str):
            raise GeometryError('Object id must be a UUID')
        try:
            identifier = str(UUID(item['id']))
        except ValueError as exc:
            raise GeometryError('Object id must be a UUID') from exc
        if identifier in identifiers:
            raise GeometryError('Object ids must be unique')
        identifiers.add(identifier)
        label = item['label']
        if not isinstance(label, str) or not 1 <= len(label) <= 80 or not label.strip():
            raise GeometryError('Object label needs 1..80 characters')
        objects.append(dict(id=identifier, label=label, polygon=_polygon(item['polygon'], 'Object polygon')))
    metric = value['metric']
    if metric is not None:
        _keys(metric, ('corners', 'width_m', 'depth_m'), 'Metric')
        if floor is None:
            raise GeometryError('Metric calibration requires a floor')
        corners = _polygon(metric['corners'], 'Metric corners')
        if len(corners) != 4 or polygon_area(corners) < 1e-4:
            raise GeometryError('Metric needs four corners and area >= 1e-4')
        turns = [_orientation(corners[i - 1], corners[i], corners[(i + 1) % 4]) for i in range(4)]
        if not (all(turn > 0 for turn in turns) or all(turn < 0 for turn in turns)):
            raise GeometryError('Metric corners must be ordered and strictly convex')
        if not all(point_in_polygon(point, floor) for point in corners):
            raise GeometryError('Metric corners must lie inside/on the floor')
        metric = dict(corners=corners,
                      width_m=_number(metric['width_m'], 'Width in metres', maximum=1000, exclusive_minimum=True),
                      depth_m=_number(metric['depth_m'], 'Depth in metres', maximum=1000, exclusive_minimum=True))
    result = dict(version=1, floor=floor, objects=objects, metric=metric)
    if len(json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) > MAX_DOCUMENT_BYTES:
        raise GeometryError('Normalized scene exceeds 128 KiB')
    build_metric_transform(result)
    return result


def _normalized_points(points):
    centre = points.mean(axis=0)
    distances = np.linalg.norm(points - centre, axis=1)
    mean_distance = distances.mean()
    if not np.isfinite(mean_distance) or mean_distance <= 0:
        raise GeometryError('Metric normalization is degenerate')
    scale = math.sqrt(2) / mean_distance
    matrix = np.array([[scale, 0, -scale * centre[0]],
                       [0, scale, -scale * centre[1]], [0, 0, 1]], dtype=np.float64)
    normalized = (points - centre) * scale
    if not np.isfinite(normalized).all() or not np.isfinite(matrix).all():
        raise GeometryError('Metric normalization is nonfinite')
    return normalized, matrix


def _project(matrix, points, *, minimum_denominator=0):
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise GeometryError('Projection needs finite Nx2 points')
    homogeneous = np.column_stack((values, np.ones(len(values)))) @ matrix.T
    denominators = homogeneous[:, 2]
    if (not np.isfinite(homogeneous).all() or np.any(denominators == 0)
            or np.any(np.abs(denominators) < minimum_denominator)):
        raise GeometryError('Projection meets a horizon')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        result = homogeneous[:, :2] / denominators[:, None]
    if not np.isfinite(result).all():
        raise GeometryError('Projection is nonfinite')
    return result


@dataclass(frozen=True, slots=True)
class MetricTransform:
    image_to_metres: np.ndarray
    metres_to_image: np.ndarray
    width_m: float
    depth_m: float

    def to_metres(self, points):
        return _project(self.image_to_metres, points, minimum_denominator=1e-6)

    def to_image(self, points):
        return _project(self.metres_to_image, points)

    def rectified_size(self):
        return rectified_size(self.width_m, self.depth_m)

    def rectify(self, frame, floor):
        """Bounded, floor-masked preview; never allocate metre-sized images."""
        height, width = frame.shape[:2]
        out_width, out_height = self.rectified_size()
        to_pixels = np.diag([(out_width - 1) / self.width_m, (out_height - 1) / self.depth_m, 1])
        from_native = np.diag([1 / width, 1 / height, 1])
        matrix = to_pixels @ self.image_to_metres @ from_native
        result = cv2.warpPerspective(frame, matrix, (out_width, out_height))
        mask = np.zeros((height, width), np.uint8)
        vertices = np.rint(np.asarray(floor) * [width, height]).astype(np.int32)
        cv2.fillPoly(mask, [vertices], 255)
        projected_mask = cv2.warpPerspective(mask, matrix, (out_width, out_height), flags=cv2.INTER_NEAREST)
        result[projected_mask == 0] = 0
        return result


def build_metric_transform(geometry):
    """Compile already structure-validated geometry, checking numerical safety.

    Cache this result by saved scene revision; do not recompute it per frame.
    H is image-normalized -> measured metres, scaled by its largest coefficient.
    """
    metric = geometry['metric']
    if metric is None:
        return None
    source = np.asarray(metric['corners'], dtype=np.float64)
    width, depth = metric['width_m'], metric['depth_m']
    destination = np.array([[0, 0], [width, 0], [width, depth], [0, depth]], dtype=np.float64)
    try:
        normalized_source, source_normalization = _normalized_points(source)
        normalized_destination, destination_normalization = _normalized_points(destination)
        normalized_h = cv2.getPerspectiveTransform(normalized_source.astype(np.float32),
                                                  normalized_destination.astype(np.float32), cv2.DECOMP_SVD)
        condition = np.linalg.cond(normalized_h)
        if not np.isfinite(normalized_h).all() or not math.isfinite(condition) or condition > 1e8:
            raise GeometryError('Metric homography is ill-conditioned')
        matrix = np.linalg.inv(destination_normalization) @ normalized_h @ source_normalization
        scale = np.max(np.abs(matrix))
        if not np.isfinite(matrix).all() or not math.isfinite(scale) or scale == 0:
            raise GeometryError('Metric homography is nonfinite or singular')
        matrix /= scale
        inverse = np.linalg.inv(matrix)
        if not np.isfinite(inverse).all():
            raise GeometryError('Metric inverse is nonfinite')
        floor = np.asarray(geometry['floor'], dtype=np.float64)
        denominators = np.column_stack((floor, np.ones(len(floor)))) @ matrix[2]
        if not (np.all(denominators >= 1e-6) or np.all(denominators <= -1e-6)):
            raise GeometryError('Metric horizon crosses or approaches the floor')
        # Verify both corner correspondences and image-space inverse roundtrip.
        mapped_corners = _project(matrix, source)
        normalized_actual = _project(destination_normalization, mapped_corners)
        if np.max(np.abs(normalized_actual - normalized_destination)) > 1e-5:
            raise GeometryError('Metric corner correspondence error exceeds 1e-5')
        all_points = np.vstack((source, floor))
        roundtrip = _project(inverse, _project(matrix, all_points))
        if np.max(np.linalg.norm(roundtrip - all_points, axis=1)) > 1e-5:
            raise GeometryError('Metric inverse roundtrip error exceeds 1e-5')
    except (cv2.error, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        raise GeometryError('Metric homography cannot be computed safely') from exc
    matrix.setflags(write=False)
    inverse.setflags(write=False)
    return MetricTransform(matrix, inverse, width, depth)


def rectified_size(width_m, depth_m):
    """Return width,height with bounded pixel count; metres determine aspect only."""
    _number(width_m, 'Width in metres', maximum=1000, exclusive_minimum=True)
    _number(depth_m, 'Depth in metres', maximum=1000, exclusive_minimum=True)
    longest = min(MAX_RECTIFIED_EDGE, math.isqrt(MAX_RECTIFIED_PIXELS))
    if width_m >= depth_m:
        return longest, max(2, round(longest * (depth_m / width_m)))
    return max(2, round(longest * (width_m / depth_m))), longest
