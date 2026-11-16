"""Bundle-backed YuNet detection and AuraFace embeddings; no model downloads."""
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.ai.model_bundle import ModelBundle, ModelBundleError
from app.ai.providers import model_providers

logger = logging.getLogger("app.ai.face")
ALIGNMENT_ID = "arcface-5pt-112-v1"
_STRIDES = (8, 16, 32)
_HEADS = ("cls", "obj", "bbox", "kps")
_OUTPUT_NAMES = [f"{head}_{stride}" for head in _HEADS for stride in _STRIDES]
_DETECTION_THRESHOLD = .5

# ArcFace five-point alignment from InsightFace's MIT-licensed code utilities:
# https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
# Copyright (c) InsightFace contributors.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE. This code license does not grant rights to upstream models.
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def norm_crop(img: np.ndarray, landmark: np.ndarray) -> np.ndarray:
    """The upstream SimilarityTransform/warpAffine alignment, fixed at 112."""
    from skimage import transform as trans

    landmark = np.asarray(landmark, dtype=np.float32)
    if (landmark.shape != (5, 2) or not np.isfinite(landmark).all()
            or np.linalg.matrix_rank(landmark - landmark.mean(axis=0)) < 2
            or np.linalg.norm(landmark[1] - landmark[0]) < 1.):
        raise ValueError("Invalid five-point face alignment")
    tform = trans.SimilarityTransform.from_estimate(landmark, _ARCFACE_DST)
    if not isinstance(tform, trans.SimilarityTransform):
        raise ValueError("Could not estimate face alignment")
    matrix = tform.params[0:2, :]
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:, :2]) <= 0:
        raise ValueError("Invalid face alignment transform")
    return cv2.warpAffine(img, matrix, (112, 112), borderValue=0.0)


@dataclass
class FaceInfo:
    # Keep these four positionals: enrollment/quality and live callers share them.
    bbox: np.ndarray
    det_score: float
    embedding: np.ndarray | None
    aligned: np.ndarray | None
    native_bbox: np.ndarray | None = None
    landmarks: np.ndarray | None = None
    model_id: str | None = None

    def __post_init__(self):
        if self.native_bbox is None:
            self.native_bbox = np.asarray(self.bbox, dtype=np.float32).copy()


class FaceEngine:
    """One owner per live/enrollment engine; request serialization is external."""

    def __init__(self, bundle: ModelBundle, providers: list, *, det_size: int = 640):
        import onnxruntime as ort

        if type(det_size) is not int or det_size < 32:
            raise ValueError("det_size must be an integer >= 32")
        detection_path = bundle.require("face_detection")
        recognition_path = bundle.require("face_recognition")
        self.model_id = bundle.face_model_id
        self.alignment_id = bundle.alignment_id
        if self.alignment_id != ALIGNMENT_ID:
            raise ModelBundleError("MODEL_CONTRACT_INVALID", "face_recognition", "Unknown alignment")
        self.det_size = det_size
        self.detection_session = ort.InferenceSession(str(detection_path), providers=model_providers(providers))
        self.recognition_session = ort.InferenceSession(str(recognition_path), providers=model_providers(providers))
        detection_inputs = self.detection_session.get_inputs()
        recognition_inputs = self.recognition_session.get_inputs()
        if (len(detection_inputs) != 1
                or {item.name for item in self.detection_session.get_outputs()} != set(_OUTPUT_NAMES)):
            raise ModelBundleError("MODEL_CONTRACT_INVALID", "face_detection", "YuNet named heads")
        recognition_outputs = self.recognition_session.get_outputs()
        if len(recognition_inputs) != 1 or len(recognition_outputs) != 1:
            raise ModelBundleError("MODEL_CONTRACT_INVALID", "face_recognition", "AuraFace IO")
        self._detection_input = detection_inputs[0].name
        self._recognition_input = recognition_inputs[0].name
        self._recognition_output = recognition_outputs[0].name
        logger.info("FaceEngine ready: model=%s, det_size=%d, providers=%s",
                    self.model_id, self.det_size, self.providers())

    def providers(self) -> dict[str, list[str]]:
        return {"detection": self.detection_session.get_providers(),
                "recognition": self.recognition_session.get_providers()}

    def detect_faces(self, img_bgr: np.ndarray) -> list[FaceInfo]:
        """Detect/align only; all boxes/landmarks and size checks remain native."""
        if (not isinstance(img_bgr, np.ndarray) or img_bgr.dtype != np.uint8
                or img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or not img_bgr.size):
            raise ValueError("Expected a nonempty uint8 BGR image")
        height, width = img_bgr.shape[:2]
        scale = self.det_size / max(height, width)
        resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(img_bgr, (resized_width, resized_height))
        padded_width, padded_height = (resized_width + 31) // 32 * 32, (resized_height + 31) // 32 * 32
        padded = cv2.copyMakeBorder(resized, 0, padded_height - resized_height,
                                    0, padded_width - resized_width, cv2.BORDER_CONSTANT, value=0)
        tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        outputs = self.detection_session.run(_OUTPUT_NAMES, {self._detection_input: tensor})
        if len(outputs) != len(_OUTPUT_NAMES):
            raise ModelBundleError("MODEL_CONTRACT_INVALID", "face_detection", "Missing YuNet heads")
        heads = dict(zip(_OUTPUT_NAMES, outputs))
        boxes, landmarks, scores = [], [], []
        for stride in _STRIDES:
            columns, rows = padded_width // stride, padded_height // stride
            count = columns * rows
            decoded = {}
            for head, channels in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)):
                values = np.asarray(heads[f"{head}_{stride}"])
                if values.shape != (1, count, channels) or not np.isfinite(values).all():
                    raise ModelBundleError("MODEL_CONTRACT_INVALID", "face_detection", "Invalid YuNet head")
                decoded[head] = values[0]
            confidence = np.sqrt(np.clip(decoded["cls"][:, 0], 0., 1.)
                                 * np.clip(decoded["obj"][:, 0], 0., 1.))
            indices = np.flatnonzero(confidence >= _DETECTION_THRESHOLD)
            if not len(indices):
                continue
            grid = np.column_stack((indices % columns, indices // columns))
            raw_boxes = decoded["bbox"][indices]
            centers = (grid + raw_boxes[:, :2]) * stride
            with np.errstate(over="ignore"):
                sizes = np.exp(raw_boxes[:, 2:]) * stride
            xyxy = np.column_stack((centers - sizes / 2., centers + sizes / 2.))
            points = (decoded["kps"][indices].reshape(-1, 5, 2) + grid[:, None]) * stride
            valid = (np.isfinite(xyxy).all(axis=1) & (sizes > 0).all(axis=1)
                     & (np.abs(xyxy) < np.iinfo(np.int32).max / 2).all(axis=1)
                     & (centers[:, 0] >= 0) & (centers[:, 0] < resized_width)
                     & (centers[:, 1] >= 0) & (centers[:, 1] < resized_height))
            boxes.extend(xyxy[valid])
            landmarks.extend(points[valid])
            scores.extend(confidence[indices][valid])
        if not boxes:
            return []
        boxes = np.asarray(boxes, dtype=np.float32)
        # OpenCV FaceDetectorYN uses integer xywh rectangles for NMS, not xyxy.
        rectangles = np.column_stack((boxes[:, :2], boxes[:, 2:] - boxes[:, :2])).astype(np.int32)
        keep = ([0] if len(boxes) == 1 else
                cv2.dnn.NMSBoxes(rectangles.tolist(), np.asarray(scores).tolist(),
                                 _DETECTION_THRESHOLD, .3, top_k=5000))
        result = []
        native_scale = np.array([width / resized_width, height / resized_height], dtype=np.float32)
        box_scale = np.tile(native_scale, 2)
        for index in np.asarray(keep).reshape(-1):
            box = boxes[index] * box_scale
            box[[0, 2]] = np.clip(box[[0, 2]], 0., width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0., height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            points = np.asarray(landmarks[index], dtype=np.float32) * native_scale
            try:
                aligned = norm_crop(img_bgr, points)
            except (ValueError, np.linalg.LinAlgError, cv2.error):
                aligned = None
            result.append(FaceInfo(box, float(scores[index]), None, aligned,
                                   native_bbox=box.copy(), landmarks=points, model_id=self.model_id))
        result.sort(key=lambda face: face.det_score, reverse=True)
        return result

    def embed_aligned(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed every aligned BGR crop, ordered and normalized; chunks <=16."""
        result = np.empty((len(crops), 512), dtype=np.float32)
        for start in range(0, len(crops), 16):
            chunk = crops[start:start + 16]
            if any(not isinstance(crop, np.ndarray) or crop.dtype != np.uint8
                   or crop.shape != (112, 112, 3) for crop in chunk):
                raise ValueError("Expected aligned uint8 112x112 BGR crops")
            tensor = np.stack(chunk)[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32)
            tensor -= 127.5
            tensor /= 127.5
            output = self.recognition_session.run([self._recognition_output],
                                                  {self._recognition_input: np.ascontiguousarray(tensor)})
            vectors = np.asarray(output[0]) if len(output) == 1 else np.empty(0)
            if vectors.shape != (len(chunk), 512) or not np.isfinite(vectors).all():
                raise ModelBundleError("MODEL_CONTRACT_INVALID", "face_recognition", "Invalid embedding batch")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise ModelBundleError("MODEL_CONTRACT_INVALID", "face_recognition", "Invalid embedding norm")
            result[start:start + len(chunk)] = vectors / norms
        return result

    def detect(self, img_bgr: np.ndarray, max_num: int = 0) -> list[FaceInfo]:
        """Real enrollment API; limit its shared-engine embedding chunks to four."""
        if type(max_num) is not int or max_num < 0:
            raise ValueError("max_num must be a nonnegative integer")
        faces = self.detect_faces(img_bgr)
        if max_num:
            faces = faces[:max_num]
        eligible = [face for face in faces if face.aligned is not None]
        for start in range(0, len(eligible), 4):
            chunk = eligible[start:start + 4]
            vectors = self.embed_aligned([face.aligned for face in chunk])
            for face, embedding in zip(chunk, vectors):
                face.embedding = embedding
        return faces
