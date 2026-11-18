"""Selected MobileSAM ONNX adapter; no downloads, semantics or fallback masks.

Preprocessing and prompt labels follow the upstream SamPredictor/SamOnnxModel
contract (MobileSAM f706ad9c4eb7f219c00d9050e46328518ffb65d2). The exported
prompt decoder already removes padding and restores native image dimensions.
SceneAssistanceService owns the one native worker and binds token validation.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from app.ai.model_bundle import ModelBundle
from app.scene_geometry import GeometryError, polygon_area, validate_geometry

CACHE_TTL = 300
MIN_FLOOR_AREA = .05


class SceneSegmentationError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PolygonSuggestion:
    points: list[list[float]]
    score: float
    needs_review: bool = True


def validate_prompts(positive_points, negative_points, box=None):
    """Return independent strict normalized original-image prompts."""
    def number(value):
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError('Prompt coordinates must be finite numbers in [0,1]')
        return float(value)

    def points(value):
        if not isinstance(value, list) or len(value) > 32:
            raise ValueError('Expected at most 32 prompt points')
        result = []
        for point in value:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError('Prompt points must contain x,y')
            result.append([number(point[0]), number(point[1])])
        return result

    positive, negative = points(positive_points), points(negative_points)
    if len(positive) + len(negative) > 32:
        raise ValueError('At most 32 positive and negative points in total')
    rectangle = None
    if box is not None:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('Prompt box must contain x1,y1,x2,y2')
        rectangle = [number(value) for value in box]
        if rectangle[0] >= rectangle[2] or rectangle[1] >= rectangle[3]:
            raise ValueError('Prompt box must have positive width and height')
    if not positive and not negative and rectangle is None:
        raise ValueError('Supply a point or a box; no automatic semantic segmentation')
    return positive, negative, rectangle


def _resize_shape(height, width):
    ratio = 1024. / max(height, width)
    return int(height * ratio + .5), int(width * ratio + .5)


def _preprocess(frame):
    if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
            or min(frame.shape[:2]) < 1):
        raise SceneSegmentationError('SCENE_INFERENCE_FAILED', 'Expected native uint8 BGR image')
    height, width = _resize_shape(*frame.shape[:2])
    # Upstream numpy apply_image uses torchvision's PIL bilinear/antialias path,
    # not OpenCV resize (different downsampling on textured photographs).
    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized = np.asarray(rgb.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    resized -= np.array([123.675, 116.28, 103.53], dtype=np.float32)
    resized /= np.array([58.395, 57.12, 57.375], dtype=np.float32)
    data = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
    data[0, :, :height, :width] = resized.transpose(2, 0, 1)
    return data


def _prompt_inputs(height, width, positive, negative, box):
    new_height, new_width = _resize_shape(height, width)
    coords = positive + negative
    labels = [1.] * len(positive) + [0.] * len(negative)
    if box is None:
        coords = coords + [[0., 0.]]
        labels.append(-1.)
    else:
        coords = coords + [box[:2], box[2:]]
        labels.extend([2., 3.])
    # normalized native * native dimensions * rounded resize/native dimensions
    # simplifies to normalized * rounded dimensions. No +.5 here: decoder owns it.
    coordinates = np.asarray(coords, dtype=np.float32)[None, :, :]
    coordinates *= np.array([new_width, new_height], dtype=np.float32)
    return dict(point_coords=coordinates, point_labels=np.asarray([labels], dtype=np.float32),
                mask_input=np.zeros((1, 1, 256, 256), dtype=np.float32),
                has_mask_input=np.zeros(1, dtype=np.float32),
                orig_im_size=np.array([height, width], dtype=np.float32))


def _mask_polygon(binary, *, diagnostics=None):
    """Trace without topology repair; optional diagnostics contain scalars only."""
    binary = np.asarray(binary, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if diagnostics is not None:
        # RETR_TREE alternates foreground boundaries and holes, including nested
        # foreground islands. Walk once, without allocating a native label image.
        components, holes = 0, 0
        pending = [(0, 0)] if hierarchy is not None and contours else []
        while pending:
            index, depth = pending.pop()
            if depth % 2:
                holes += 1
            else:
                components += 1
            following, _previous, child, _parent = hierarchy[0, index]
            if following >= 0:
                pending.append((int(following), depth))
            if child >= 0:
                pending.append((int(child), depth + 1))
        diagnostics.update(foreground_pixels=int(np.count_nonzero(binary)),
                           foreground_components_8=components, hole_count=holes,
                           contour_count=len(contours), positive_area_components_4=None,
                           polygon_valid=False, vertex_count=None, rejection_reason=None,
                           simplification_attempts=[])
    # A single simple polygon cannot represent holes or disconnected components.
    # Never silently fill them, bridge desks or substitute a bounding rectangle.
    if len(contours) != 1 or hierarchy is None:
        if diagnostics is not None:
            diagnostics['rejection_reason'] = ('NO_CONTOURS' if not contours else
                                               'MULTIPLE_CONTOURS' if len(contours) != 1 else
                                               'MISSING_CONTOUR_HIERARCHY')
        return None
    if cv2.contourArea(contours[0]) <= 0:
        if diagnostics is not None:
            diagnostics['rejection_reason'] = 'ZERO_CONTOUR_AREA'
        return None
    # Contours trace pixel centres, not pixel cells. Moore tracing can turn two
    # regions sharing just one foreground sample into a simple diagonal bridge
    # with no repeated contour vertex. Check raw positive-area support FIRST:
    # each full 2x2 block spans a unit image-space cell; those cells must connect
    # through shared edges, not merely through a point/diagonal. A real two-pixel
    # neck remains connected, as do genuine connected concave regions.
    area_cells = binary[:-1, :-1] & binary[1:, :-1] & binary[:-1, 1:] & binary[1:, 1:]
    area_components = cv2.connectedComponents(area_cells, connectivity=4)[0] - 1
    if diagnostics is not None:
        diagnostics['positive_area_components_4'] = area_components
    if area_components > 1:
        if diagnostics is not None:
            diagnostics['rejection_reason'] = 'DISCONNECTED_POSITIVE_AREA_SUPPORT'
        return None
    contour = contours[0]
    if len(np.unique(contour[:, 0, :], axis=0)) != len(contour):
        # Point-touching lobes retrace vertices; do not smooth that invalid
        # topology into an apparently valid outline during simplification.
        if diagnostics is not None:
            diagnostics['rejection_reason'] = 'REPEATED_CONTOUR_VERTICES'
        return None
    perimeter = cv2.arcLength(contour, True)
    for fraction in (0., .0005, .001, .002, .004, .008):
        simplified = cv2.approxPolyDP(contour, fraction * perimeter, True)
        if diagnostics is not None:
            attempt = dict(epsilon_fraction=fraction, vertex_count=len(simplified), rejection_reason=None)
            diagnostics['simplification_attempts'].append(attempt)
        if len(simplified) > 128:
            if diagnostics is not None:
                attempt['rejection_reason'] = 'VERTEX_BUDGET_EXCEEDED'
            continue
        if len(simplified) < 3:
            if diagnostics is not None:
                diagnostics['rejection_reason'] = attempt['rejection_reason'] = 'TOO_FEW_SIMPLIFIED_VERTICES'
            return None
        height, width = binary.shape
        points = [[float(x) / width, float(y) / height] for x, y in simplified[:, 0, :]]
        try:
            validate_geometry(dict(version=2, floor=points, objects=[], doors=[], metric=None))
        except GeometryError as exc:
            if diagnostics is not None:
                diagnostics['rejection_reason'] = attempt['rejection_reason'] = 'INVALID_SCENE_POLYGON'
                diagnostics['geometry_error'] = str(exc)
            return None
        if diagnostics is not None:
            diagnostics.update(polygon_valid=True, vertex_count=len(points))
        return points
    if diagnostics is not None:
        diagnostics['rejection_reason'] = 'VERTEX_BUDGET_EXCEEDED'
    return None


def _floor_polygon(binary):
    """A floor is one region: drop specks and fill furniture holes, which suggestions never do.

    Only floors are repaired this way; object and prompted outlines still go through _mask_polygon unchanged.
    """
    binary = np.asarray(binary, dtype=np.uint8)
    height, width = binary.shape
    kernel_size = max(3, round(.01 * min(height, width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def largest(mask):
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count < 2:
            return None
        return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

    region = largest(binary)
    if region is None:
        return None
    region = cv2.morphologyEx(cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    region = largest(region)
    if region is None:
        return None
    contours, _hierarchy = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(region)
    cv2.fillPoly(filled, [max(contours, key=cv2.contourArea)], 1)
    return _mask_polygon(filled)


def _best_floor(masks, scores):
    if (not isinstance(masks, np.ndarray) or masks.ndim != 4 or masks.shape[:2] != (1, 4)
            or masks.dtype != np.float32 or not isinstance(scores, np.ndarray)
            or scores.shape != (1, 4) or scores.dtype != np.float32
            or not np.isfinite(masks).all() or not np.isfinite(scores).all()):
        raise SceneSegmentationError('SCENE_INFERENCE_FAILED', 'MobileSAM output contract mismatch')
    for index in np.argsort(-scores[0], kind='stable'):
        points = _floor_polygon(masks[0, index] > 0.)
        if points is not None and polygon_area(points) >= MIN_FLOOR_AREA:
            return PolygonSuggestion(points, float(scores[0, index]))
    return None


def _rank_polygons(masks, scores, height, width, *, diagnostics=None):
    if (not isinstance(masks, np.ndarray) or masks.shape != (1, 4, height, width)
            or masks.dtype != np.float32 or not isinstance(scores, np.ndarray)
            or scores.shape != (1, 4) or scores.dtype != np.float32
            or not np.isfinite(masks).all() or not np.isfinite(scores).all()):
        raise SceneSegmentationError('SCENE_INFERENCE_FAILED', 'MobileSAM output contract mismatch')
    result = []
    for index in np.argsort(-scores[0], kind='stable'):
        detail = None
        if diagnostics is not None:
            detail = dict(mask_index=int(index), native_score=float(scores[0, index]),
                          native_logit_min=float(masks[0, index].min()),
                          native_logit_max=float(masks[0, index].max()), selected=False)
            diagnostics.append(detail)
        points = _mask_polygon(masks[0, index] > 0., diagnostics=detail)
        if points is not None and len(result) < 3:
            result.append(PolygonSuggestion(points, float(scores[0, index])))
            if detail is not None:
                detail['selected'] = True
            if len(result) == 3 and diagnostics is None:
                break
    return result


@dataclass(slots=True)
class _Embedding:
    camera_id: int
    snapshot_id: str
    expires_mono: float
    value: np.ndarray


class SceneSegmenter:
    def __init__(self, bundle: ModelBundle, *, monotonic=time.monotonic):
        self._monotonic = monotonic
        self._cache_lock = threading.Lock()
        self._native_lock = threading.Lock()
        self._cache = OrderedDict()
        self._epochs = {}
        self._validator = None
        self.last_inference_seconds = None
        try:
            encoder = bundle.require('scene')
            decoder = bundle.component_paths['scene_decoder']
            declaration = bundle.metadata['components']['scene']
            canonical = dict(preprocessing_id=declaration['preprocessing_id'],
                             files={item['role']: item['sha256'] for item in declaration['files']})
            self.model_id = hashlib.sha256(json.dumps(canonical, sort_keys=True,
                                                      separators=(',', ':')).encode()).hexdigest()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._encoder = ort.InferenceSession(str(encoder), sess_options=options,
                                                 providers=['CPUExecutionProvider'])
            self._decoder = ort.InferenceSession(str(decoder), sess_options=options,
                                                 providers=['CPUExecutionProvider'])
            if any(value != ['CPUExecutionProvider'] for value in self.providers().values()):
                raise ValueError('Scene inference must be CPU only')
            encoder_inputs, encoder_outputs = self._encoder.get_inputs(), self._encoder.get_outputs()
            if len(encoder_inputs) != 1 or len(encoder_outputs) != 1:
                raise ValueError('Invalid encoder IO')
            self._image_input = encoder_inputs[0].name
            self._embedding_output = encoder_outputs[0].name
            if {item.name for item in self._decoder.get_inputs()} != {
                    'image_embeddings', 'point_coords', 'point_labels', 'mask_input', 'has_mask_input', 'orig_im_size'}:
                raise ValueError('Invalid prompt decoder inputs')
            if {item.name for item in self._decoder.get_outputs()} != {'masks', 'iou_predictions', 'low_res_masks'}:
                raise ValueError('Invalid prompt decoder outputs')
        except Exception as exc:
            raise SceneSegmentationError('SCENE_MODEL_UNAVAILABLE', 'Selected MobileSAM component is unavailable') from exc

    def providers(self):
        return {'encoder': self._encoder.get_providers(), 'decoder': self._decoder.get_providers()}

    def set_snapshot_validator(self, callback):
        """Bind SceneService.get_snapshot once, before requests are admitted."""
        self._validator = callback

    def invalidate(self, camera_id):
        with self._cache_lock:
            self._epochs[camera_id] = self._epochs.get(camera_id, 0) + 1
            self._cache = OrderedDict((key, item) for key, item in self._cache.items()
                                      if item.camera_id != camera_id)

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()
            # Also prevent an in-progress encoder from publishing after close.
            for camera in self._epochs:
                self._epochs[camera] += 1

    def purge_expired(self):
        """Release expired embeddings; called by the owner's idle maintenance."""
        with self._cache_lock:
            now = self._monotonic()
            self._cache = OrderedDict((key, item) for key, item in self._cache.items()
                                      if now < item.expires_mono)

    def _check(self, snapshot, epoch):
        if self._monotonic() >= snapshot.expires_mono:
            raise SceneSegmentationError('SCENE_SNAPSHOT_EXPIRED', 'Frozen snapshot expired')
        if self._validator is None:
            raise SceneSegmentationError('SCENE_INFERENCE_FAILED', 'Snapshot validator is not bound')
        self._validator(snapshot.snapshot_id, snapshot.camera_id)
        with self._cache_lock:
            if self._epochs.get(snapshot.camera_id, 0) != epoch:
                raise SceneSegmentationError('CAMERA_SOURCE_CHANGED', 'Snapshot invalidated during inference')

    def suggest(self, snapshot, positive_points, negative_points, box=None):
        return self._infer(snapshot, positive_points, negative_points, box,
                           lambda masks, scores: _rank_polygons(masks, scores, snapshot.height, snapshot.width))

    def floor(self, snapshot, positive_points, negative_points):
        """First-ranked usable floor outline, repaired for furniture holes, or None."""
        return self._infer(snapshot, positive_points, negative_points, None, _best_floor)

    def _infer(self, snapshot, positive_points, negative_points, box, convert):
        positive, negative, rectangle = validate_prompts(positive_points, negative_points, box)
        if not self._native_lock.acquire(blocking=False):
            raise SceneSegmentationError('SCENE_BUSY', 'Scene native worker is occupied')
        started = self._monotonic()
        try:
            self.purge_expired()
            with self._cache_lock:
                epoch = self._epochs.setdefault(snapshot.camera_id, 0)
                retained = tuple((key, item.snapshot_id, item.camera_id) for key, item in self._cache.items())
            # Database/reader checks never run under cache or native-publication locks.
            for key, token, camera in retained:
                try:
                    if self._validator is not None:
                        self._validator(token, camera)
                except Exception:
                    with self._cache_lock:
                        self._cache.pop(key, None)
            self._check(snapshot, epoch)
            if snapshot.packet.frame.shape[:2] != (snapshot.height, snapshot.width):
                raise SceneSegmentationError('SCENE_INFERENCE_FAILED', 'Snapshot dimensions differ from metadata')
            key = (self.model_id, snapshot.snapshot_id, snapshot.camera_id,
                   snapshot.packet.source_revision, snapshot.packet.stream_id, snapshot.packet.seq,
                   snapshot.width, snapshot.height)
            with self._cache_lock:
                cached = self._cache.get(key)
            if cached is None:
                embedding = self._encoder.run([self._embedding_output],
                                              {self._image_input: _preprocess(snapshot.packet.frame)})[0]
                if (embedding.shape != (1, 256, 64, 64) or embedding.dtype != np.float32
                        or not np.isfinite(embedding).all()):
                    raise SceneSegmentationError('SCENE_INFERENCE_FAILED', 'MobileSAM encoder output contract mismatch')
                self._check(snapshot, epoch)
                with self._cache_lock:
                    if (self._epochs.get(snapshot.camera_id, 0) != epoch
                            or self._monotonic() >= snapshot.expires_mono):
                        raise SceneSegmentationError('CAMERA_SOURCE_CHANGED', 'Snapshot retired before cache publication')
                    while len(self._cache) >= 2:
                        self._cache.popitem(last=False)
                    self._cache[key] = _Embedding(snapshot.camera_id, snapshot.snapshot_id,
                                                  min(snapshot.expires_mono, self._monotonic() + CACHE_TTL), embedding)
            else:
                embedding = cached.value
            self._check(snapshot, epoch)
            feeds = _prompt_inputs(snapshot.height, snapshot.width, positive, negative, rectangle)
            feeds['image_embeddings'] = embedding
            masks, scores = self._decoder.run(['masks', 'iou_predictions'], feeds)
            self._check(snapshot, epoch)
            result = convert(masks, scores)
            self._check(snapshot, epoch)
            return result
        except Exception:
            # Reader replacement may be discovered before publication callbacks.
            # Discard that token's cached embedding as well as its stale result.
            with self._cache_lock:
                self._cache = OrderedDict((key, item) for key, item in self._cache.items()
                                          if item.snapshot_id != snapshot.snapshot_id)
            raise
        finally:
            # The cache stores only embeddings, never snapshots/readers/native frames.
            self.last_inference_seconds = max(0., self._monotonic() - started)
            self._native_lock.release()
