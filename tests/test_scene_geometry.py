"""Authoritative scene v2 geometry; isolated, no cameras or model sessions.

Run: .venv/Scripts/python.exe tests/test_scene_geometry.py
Execution belongs to the integration owner.
"""
import copy
import math
import sys
import traceback
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.scene_geometry import (
    GeometryError, build_metric_transform, point_in_polygon, rectified_size,
    validate_door_quad, validate_geometry,
)

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def geometry(*, metric=False):
    return dict(version=2, floor=[[0, 0], [1, 0], [1, 1], [0, 1]], objects=[], doors=[],
                metric=dict(corners=[[.2, .2], [.8, .2], [.8, .8], [.2, .8]],
                            width_m=4, depth_m=3) if metric else None)


def rejects(value):
    try:
        validate_geometry(value)
    except GeometryError:
        return
    raise AssertionError(f"Accepted invalid scene: {value!r}")


@test
def test_preserves_native_coordinate_precision_and_concave_floor():
    value = geometry()
    value['floor'] = [[.123456789012345, 0], [1, 0], [1, .4], [.5, .4], [.5, 1], [0, 1]]
    value['objects'] = [dict(id=str(uuid4()), label='Стол', polygon=[[.1, .1], [.2, .1], [.2, .2]])]
    result = validate_geometry(value)
    assert result == value and result is not value
    assert point_in_polygon([.3, .8], result['floor'])
    assert not point_in_polygon([.8, .8], result['floor'])
    assert point_in_polygon([.5, .8], result['floor'])
    result['floor'][0][0] = 0
    assert value['floor'][0][0] == .123456789012345
    assert validate_geometry(dict(version=2, floor=None, objects=[], doors=[], metric=None))['floor'] is None
    # A nonzero polygon must not become \"zero area\" through float underflow.
    tiny = [[0, 0], [1e-200, 0], [0, 1e-200]]
    assert validate_geometry(dict(geometry(), floor=tiny))['floor'] == tiny


@test
def test_rejects_unknown_keys_and_non_json_numeric_abuse():
    for key in ('version', 'floor', 'objects', 'doors', 'metric'):
        value = geometry()
        del value[key]
        rejects(value)
    rejects(dict(geometry(), extra=True))
    rejects(dict(geometry(), version=True))
    rejects(dict(geometry(), version=2.0))
    rejects(dict(geometry(), version=1))
    for number in (True, False, float('nan'), float('inf'), -float('inf'), -.01, 1.01, '0.5', None, 10 ** 1000):
        value = geometry()
        value['floor'][0][0] = number
        rejects(value)
    for value in (None, [], 'scene', dict(geometry(), objects={})):
        rejects(value)
    obj = dict(id=str(uuid4()), label='Стол', polygon=[[0, 0], [1, 0], [1, 1]])
    for changed in (dict(obj, extra=1), dict(obj, id='bad'), dict(obj, label=''),
                    dict(obj, label=' '), dict(obj, label='x' * 81), dict(obj, label=3)):
        rejects(dict(geometry(), objects=[changed]))
    rejects(dict(geometry(), metric={'corners': [], 'width_m': 1, 'depth_m': 1, 'extra': 0}))


@test
def test_polygon_distinctness_area_intersection_and_limits():
    for polygon in (
        [], [[0, 0], [1, 1]], [[0, 0], [1, 0], [0, 0]],
        [[0, 0], [.5, .5], [1, 1]],
        [[0, 0], [1, 1], [0, 1], [1, 0]],
        [[0, 0], [1, 0], [.5, 0], [1, 1], [0, 1]],
        [[0, 0], [1, 0], [1, 1], [.5, 0], [0, 1]],
        [[0, 0, 0], [1, 0], [1, 1]],
    ):
        rejects(dict(geometry(), floor=polygon))
    circle = [[.5 + .4 * math.cos(i * 2 * math.pi / 128),
               .5 + .4 * math.sin(i * 2 * math.pi / 128)] for i in range(128)]
    assert len(validate_geometry(dict(geometry(), floor=circle))['floor']) == 128
    rejects(dict(geometry(), floor=circle + [[.95, .5]]))
    objects = [dict(id=str(uuid4()), label='Object', polygon=[[0, 0], [1, 0], [1, 1]]) for _ in range(32)]
    assert len(validate_geometry(dict(geometry(), objects=objects))['objects']) == 32
    rejects(dict(geometry(), objects=objects + [dict(objects[0], id=str(uuid4()))]))
    rejects(dict(geometry(), objects=[objects[0], objects[0]]))
    rejects(dict(geometry(), objects=[objects[0], dict(objects[0], id=objects[0]['id'].upper())]))
    huge_objects = [dict(id=str(uuid4()), label='x' * 80, polygon=circle) for _ in range(32)]
    rejects(dict(geometry(), objects=huge_objects))


def door(**changes):
    return dict(dict(id=str(uuid4()), label='Дверь', kind='DOOR',
                     quad=[[.4, .8], [.6, .8], [.6, .2], [.4, .2]]), **changes)


@test
def test_door_is_validated_and_copied_without_rounding():
    value = dict(geometry(), doors=[door(quad=[[.1234567890123, .8], [.6, .8], [.6, .2], [.1234567890123, .2]],
                                         kind='PASSAGE')])
    result = validate_geometry(value)
    assert result == value and result['doors'][0]['quad'] is not value['doors'][0]['quad']
    # A door seen edge-on has a zero-area quad but is still valid.
    edge_on = door(quad=[[.5, .8], [.5, .7], [.5, .1], [.5, .2]])
    assert len(validate_geometry(dict(geometry(), doors=[edge_on]))['doors']) == 1


@test
def test_door_rejects_inverted_posts_short_threshold_and_bad_shape():
    for quad in (
        [[.4, .8], [.6, .8], [.6, .9], [.4, .2]],   # right post top below its bottom
        [[.4, .8], [.6, .8], [.6, .8], [.4, .2]],   # top level with bottom
        [[.4, .8], [.41, .8], [.41, .2], [.4, .2]],  # threshold shorter than .02
        [[.4, .8], [.6, .8], [.6, .2]],              # three points
        [[.4, .8], [.6, .8], [.6, .2], [.4, 1.2]],   # outside the frame
    ):
        rejects(dict(geometry(), doors=[door(quad=quad)]))
    for changed in (door(kind='GATE'), door(label=''), door(label='x' * 81), door(id='bad'), dict(door(), extra=1)):
        rejects(dict(geometry(), doors=[changed]))
    rejects(dict(geometry(), doors=[door() for _ in range(9)]))
    assert len(validate_geometry(dict(geometry(), doors=[door() for _ in range(8)]))['doors']) == 8


@test
def test_ids_are_unique_across_objects_and_doors():
    shared = str(uuid4())
    obj = dict(id=shared, label='Стол', polygon=[[0, 0], [1, 0], [1, 1]])
    rejects(dict(geometry(), objects=[obj], doors=[door(id=shared)]))
    rejects(dict(geometry(), doors=[door(id=shared), door(id=shared.upper())]))


@test
def test_ordered_perspective_rectangle_maps_to_measured_metres():
    value = geometry(metric=True)
    value['metric']['corners'] = [[.2, .2], [.8, .2], [1, .8], [0, .8]]
    saved = validate_geometry(value)
    transform = build_metric_transform(saved)
    # Hand-derived trapezoid: halfway down the metric depth projects to y=.425.
    np.testing.assert_allclose(transform.to_metres([[.5, .425]]), [[2, 1.5]], atol=1e-5)
    np.testing.assert_allclose(transform.to_image([[2, 1.5]]), [[.5, .425]], atol=1e-5)
    np.testing.assert_allclose(transform.to_metres(value['metric']['corners']),
                               [[0, 0], [4, 0], [4, 3], [0, 3]], atol=1e-5)
    assert np.max(np.abs(transform.image_to_metres)) == 1
    assert not transform.image_to_metres.flags.writeable
    assert build_metric_transform(geometry()) is None


@test
def test_metric_requires_convex_ordered_contained_rectangle_and_dimension_limits():
    value = geometry(metric=True)
    for corners in (
        [[.2, .2], [.8, .8], [.8, .2], [.2, .8]],
        [[.2, .2], [.8, .2], [.4, .4], [.2, .8]],
        [[.2, .2], [.8, .2], [.8, .2], [.2, .8]],
        [[.2, .2], [.21, .2], [.21, .20001], [.2, .20001]],
        [[.2, .2], [.8, .2], [.8, .8]],
    ):
        bad = copy.deepcopy(value)
        bad['metric']['corners'] = corners
        rejects(bad)
    for dimension in (0, -1, 1000.001, True, float('nan'), float('inf'), '4'):
        bad = copy.deepcopy(value)
        bad['metric']['width_m'] = dimension
        rejects(bad)
    assert validate_geometry(dict(value, metric=dict(value['metric'], width_m=1000)))['metric']['width_m'] == 1000
    rejects(dict(value, floor=None))
    concave = [[0, 0], [1, 0], [1, .4], [.5, .4], [.5, 1], [0, 1]]
    rejects(dict(value, floor=concave))


@test
def test_horizon_at_one_floor_corner_and_near_horizon_rejected():
    # For these calibration corners the projective horizon is y=.1.
    # The rectangle's denominators are all positive, but one FLOOR vertex is not.
    value = geometry(metric=True)
    value['metric']['corners'] = [[.4, .2], [.6, .2], [.9, .5], [.1, .5]]
    value['floor'] = [[.5, .05], [.8, .2], [1, .6], [0, .6], [.2, .2]]
    rejects(value)
    value['floor'] = [[.5, .100000001], [.8, .2], [1, .6], [0, .6], [.2, .2]]
    rejects(value)
    value['floor'] = [[.5, .15], [.8, .2], [1, .6], [0, .6], [.2, .2]]
    assert validate_geometry(value)['metric'] is not None


@test
def test_normalized_condition_and_inverse_roundtrip_are_enforced():
    value = geometry(metric=True)
    with patch('app.scene_geometry.np.linalg.cond', return_value=1.000001e8):
        rejects(value)
    with patch('app.scene_geometry.np.linalg.inv', return_value=np.eye(3)):
        rejects(value)
    # Strong anisotropy cannot be hidden by a successful matrix inversion.
    value['metric']['width_m'] = 1000
    value['metric']['depth_m'] = 1e-12
    rejects(value)


@test
def test_rectification_is_pixel_bounded_not_allocated_from_metres():
    for width, depth in ((1000, 1000), (1000, .001), (.001, 1000), (4, 3)):
        w, h = rectified_size(width, depth)
        assert 1 <= w <= 1280 and 1 <= h <= 1280 and w * h <= 1_700_000
    assert rectified_size(4, 3) == (1280, 960)
    assert rectified_size(1000, 1000) == (1280, 1280)
    value = geometry(metric=True)
    value['metric']['width_m'] = 1000
    transform = build_metric_transform(validate_geometry(value))
    frame = np.zeros((120, 160, 3), np.uint8)
    output = transform.rectify(frame, value['floor'])
    assert max(output.shape[:2]) <= 1280 and output.shape[0] * output.shape[1] <= 1_700_000
    assert frame.shape == (120, 160, 3)


if __name__ == '__main__':
    failures = 0
    for case in TESTS:
        try:
            case()
            print(f'PASS {case.__name__}')
        except Exception:
            failures += 1
            print(f'FAIL {case.__name__}')
            traceback.print_exc()
    print(f'{len(TESTS) - failures}/{len(TESTS)} scene geometry cases passed')
    raise SystemExit(bool(failures))
