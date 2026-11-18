"""One-click scene analysis orchestration with a fake detector and segmenter.

Run: .venv/Scripts/python.exe tests/test_scene_analysis.py
No model, camera or database: the contract under test is how detections become a valid
reviewable scene draft (floor prompts, object outlines, door frames, caps and warnings).
"""
import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.scene_analyzer import MAX_DOORS, MAX_OBJECTS, analyze
from app.ai.scene_detector import SceneDetection
from app.ai.scene_segmenter import PolygonSuggestion, SceneSegmentationError
from app.scene_geometry import validate_geometry

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def det(kind, box, score=.9, label=None):
    return SceneDetection(0, kind, label or kind.title(), list(box), score)


class FakeDetector:
    def __init__(self, detections):
        self.detections, self.frames = detections, []

    def detect(self, frame):
        self.frames.append(frame)
        return list(self.detections)


class FakeSegmenter:
    def __init__(self, *, floor=None, mask=None):
        self.floor_result, self.mask_result = floor, mask
        self.floor_calls, self.suggest_calls = [], []

    def floor(self, snapshot, positive, negative):
        self.floor_calls.append((positive, negative))
        return self.floor_result

    def suggest(self, snapshot, positive, negative, box):
        self.suggest_calls.append((positive, negative, box))
        return [] if self.mask_result is None else [self.mask_result]


SNAPSHOT = SimpleNamespace(packet=SimpleNamespace(frame="frame"))
FLOOR = PolygonSuggestion([[0., .6], [1., .6], [1., 1.], [0., 1.]], .93)


def inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


@test
def floor_prompts_avoid_things_standing_on_the_floor_but_not_corridors():
    desk = det("OBJECT", [.1, .7, .45, .98], .8)
    person = det("PERSON", [.55, .6, .75, 1.], .95)
    door = det("DOOR", [.45, .1, .55, .75], .7)
    corridor = det("PASSAGE", [0., .5, 1., 1.], .6, "Коридор")
    segmenter = FakeSegmenter(floor=FLOOR)
    draft = analyze(SNAPSHOT, FakeDetector([person, desk, door, corridor]), segmenter)
    (positive, negative), = segmenter.floor_calls
    assert positive and not any(inside(p, box.box) for p in positive for box in (desk, person, door))
    assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in positive)
    assert negative == [[.65, .8], [.275, .84]], "person (.95) first, then desk (.8); doors are not negatives"
    assert draft["floor"] == FLOOR.points and "FLOOR_NOT_FOUND" not in draft["warnings"]


@test
def a_scene_with_no_clear_floor_reports_it_without_failing():
    everything = det("OBJECT", [0., .6, 1., 1.])
    segmenter = FakeSegmenter(floor=FLOOR)
    draft = analyze(SNAPSHOT, FakeDetector([everything]), segmenter)
    assert draft["floor"] is None and "FLOOR_NOT_FOUND" in draft["warnings"]
    assert segmenter.floor_calls == [], "No candidate point may be asked of the model"
    draft = analyze(SNAPSHOT, FakeDetector([]), FakeSegmenter(floor=None))
    assert draft["floor"] is None
    assert draft["warnings"] == ["FLOOR_NOT_FOUND", "NO_DOORS_FOUND", "NO_OBJECTS_FOUND"]
    assert draft["objects"] == [] and draft["doors"] == []


@test
def objects_use_mask_outlines_and_fall_back_to_their_box():
    desk = det("OBJECT", [.2, .65, .5, .9], .8, "Стол")
    mask = PolygonSuggestion([[.2, .65], [.5, .65], [.45, .9]], .77)
    draft = analyze(SNAPSHOT, FakeDetector([desk]), FakeSegmenter(floor=FLOOR, mask=mask))
    item, = draft["objects"]
    assert item["outline"] == "MASK" and item["polygon"] == mask.points and item["label"] == "Стол"
    assert item["score"] == .8
    segmenter = FakeSegmenter(floor=FLOOR)
    draft = analyze(SNAPSHOT, FakeDetector([desk]), segmenter)
    item, = draft["objects"]
    assert item["outline"] == "BOX" and item["polygon"] == [[.2, .65], [.5, .65], [.5, .9], [.2, .9]]
    assert segmenter.suggest_calls == [([], [], [.2, .65, .5, .9])]


@test
def door_boxes_become_threshold_and_posts():
    door = det("DOOR", [.4, .2, .6, .8], .9, "Дверь")
    passage = det("PASSAGE", [.1, .3, .3, .9], .6, "Лестница")
    sliver = det("DOOR", [.7, .2, .71, .8], .5)
    draft = analyze(SNAPSHOT, FakeDetector([door, passage, sliver]), FakeSegmenter(floor=FLOOR))
    assert [(d["kind"], d["label"], d["quad"]) for d in draft["doors"]] == [
        ("DOOR", "Дверь", [[.4, .8], [.6, .8], [.6, .2], [.4, .2]]),
        ("PASSAGE", "Лестница", [[.1, .9], [.3, .9], [.3, .3], [.1, .3]]),
    ], "A door narrower than the minimum threshold is dropped, not stretched"
    assert "NO_DOORS_FOUND" not in draft["warnings"]


@test
def objects_and_doors_are_capped_by_score_order():
    objects = [det("OBJECT", [.05 + .03 * (i % 10), .65 + .03 * (i // 10), .08 + .03 * (i % 10),
                              .68 + .03 * (i // 10)], 1 - i / 100) for i in range(30)]
    doors = [det("DOOR", [.05 + .08 * i, .1, .09 + .08 * i, .5], 1 - i / 100) for i in range(12)]
    draft = analyze(SNAPSHOT, FakeDetector(objects + doors), FakeSegmenter(floor=FLOOR))
    assert len(draft["objects"]) == MAX_OBJECTS == 24 and len(draft["doors"]) == MAX_DOORS == 8
    assert [round(item["score"], 2) for item in draft["objects"]] == [round(1 - i / 100, 2) for i in range(24)]


@test
def the_draft_is_a_valid_scene_with_unique_identifiers_and_never_an_invalid_one():
    objects = [det("OBJECT", [.1 + .1 * i, .7, .18 + .1 * i, .9], .9 - i / 100) for i in range(5)]
    doors = [det("DOOR", [.5, .1, .7, .6])]
    draft = analyze(SNAPSHOT, FakeDetector(objects + doors), FakeSegmenter(floor=FLOOR))
    scene = validate_geometry(dict(
        version=2, floor=draft["floor"], metric=None,
        objects=[dict(id=o["id"], label=o["label"], polygon=o["polygon"]) for o in draft["objects"]],
        doors=[dict(id=d["id"], label=d["label"], kind=d["kind"], quad=d["quad"]) for d in draft["doors"]]))
    assert len(scene["objects"]) == 5 and len(scene["doors"]) == 1
    # An unusable model floor (not a simple polygon) is reported, never returned.
    bowtie = PolygonSuggestion([[0., 0.], [1., 1.], [0., 1.], [1., 0.]], .9)
    assert analyze(SNAPSHOT, FakeDetector([]), FakeSegmenter(floor=bowtie))["floor"] is None


@test
def worst_case_outlines_are_rounded_to_subpixel_precision_and_always_fit_one_document():
    angles = [i * 2 * math.pi / 128 for i in range(128)]
    ring = PolygonSuggestion([[.5 + .4 * math.cos(a), .5 + .4 * math.sin(a)] for a in angles], .5)
    objects = [det("OBJECT", [.1, .1, .9, .9], 1 - i / 100) for i in range(MAX_OBJECTS)]
    draft = analyze(SNAPSHOT, FakeDetector(objects), FakeSegmenter(floor=ring, mask=ring))
    assert len(draft["objects"]) == MAX_OBJECTS and len(draft["floor"]) == 128
    assert all(len(repr(value).split(".")[1]) <= 6 for polygon in (draft["floor"], draft["objects"][0]["polygon"])
               for point in polygon for value in point)
    validate_geometry(dict(version=2, floor=draft["floor"], metric=None, doors=[],
                           objects=[dict(id=o["id"], label=o["label"], polygon=o["polygon"]) for o in draft["objects"]]))


@test
def native_failures_and_bad_detector_output_fail_the_whole_analysis():
    class Failing(FakeSegmenter):
        def floor(self, *args):
            raise SceneSegmentationError("CAMERA_SOURCE_CHANGED", "retired")

    try:
        analyze(SNAPSHOT, FakeDetector([]), Failing())
    except SceneSegmentationError as exc:
        assert exc.code == "CAMERA_SOURCE_CHANGED"
    else:
        raise AssertionError("A retired snapshot must abort the analysis")


if __name__ == "__main__":
    failed = 0
    for case in TESTS:
        try:
            case()
            print(f"PASS {case.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {case.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(bool(failed))
