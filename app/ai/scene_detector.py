"""Fixed-vocabulary OWLv2 inference; only a declared local ONNX is loaded."""
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from app.ai.providers import model_providers
from app.ai.scene_segmenter import SceneSegmentationError
from app.ai.scene_vocabulary import VOCABULARY, VOCABULARY_ID

_CLIP_MEAN = np.array((.48145466, .4578275, .40821073), dtype=np.float32)
_CLIP_STD = np.array((.26862954, .26130258, .27577711), dtype=np.float32)
_DOOR_LOGIT_THRESHOLD = np.float32(np.log(.15 / .85))
_OBJECT_LOGIT_THRESHOLD = np.float32(np.log(.20 / .80))


@dataclass(frozen=True, slots=True)
class SceneDetection:
    prompt_index: int
    kind: str
    label: str
    box: list[float]
    score: float


class SceneObjectDetector:
    def __init__(self, bundle, providers):
        try:
            self.session = ort.InferenceSession(
                str(bundle.require("scene_detector")), providers=model_providers(providers))
            if [item.name for item in self.session.get_inputs()] != ["pixel_values"]:
                raise ValueError("Invalid OWLv2 image input")
            if {item.name for item in self.session.get_outputs()} != {"logits", "boxes"}:
                raise ValueError("Invalid OWLv2 prediction outputs")
            metadata = self.session.get_modelmeta().custom_metadata_map
            if metadata.get("na_meste.scene_vocabulary") != VOCABULARY_ID:
                raise ValueError("OWLv2 vocabulary does not match the application")
        except Exception as exc:
            raise SceneSegmentationError(
                "SCENE_DETECTOR_UNAVAILABLE", "Selected OWLv2 component is unavailable") from exc

    def detect(self, frame_bgr) -> list[SceneDetection]:
        height, width = frame_bgr.shape[:2]
        side = max(height, width)
        square = np.full((side, side, 3), 127, dtype=np.uint8)
        square[:height, :width] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(square, (960, 960), interpolation=cv2.INTER_AREA).astype(np.float32)
        image /= 255
        image -= _CLIP_MEAN
        image /= _CLIP_STD
        pixels = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        logits, boxes = self.session.run(["logits", "boxes"], {"pixel_values": pixels})
        if (logits.shape != (1, 3600, len(VOCABULARY)) or boxes.shape != (1, 3600, 4)
                or not np.isfinite(logits).all() or not np.isfinite(boxes).all()):
            raise SceneSegmentationError("SCENE_INFERENCE_FAILED", "Invalid OWLv2 prediction tensors")
        prompts = logits[0].argmax(axis=1)
        # sigmoid(max(logits)) equals max(sigmoid(logits)); compute only the winners.
        best_logits = logits[0, np.arange(len(prompts)), prompts]
        scores = np.exp(-np.logaddexp(0, -best_logits))
        groups = {"DOOR": [], "OBJECT": [], "PERSON": []}
        for patch, (prompt, score) in enumerate(zip(prompts, scores)):
            _, kind, label = VOCABULARY[int(prompt)]
            # Compare logits to avoid a rounded sigmoid excluding the boundary.
            threshold = _DOOR_LOGIT_THRESHOLD if kind in ("DOOR", "PASSAGE") else _OBJECT_LOGIT_THRESHOLD
            if best_logits[patch] < threshold:
                continue
            cx, cy, box_width, box_height = boxes[0, patch]
            x1, x2 = np.clip([(cx - box_width / 2) * side, (cx + box_width / 2) * side], 0, width) / width
            y1, y2 = np.clip([(cy - box_height / 2) * side, (cy + box_height / 2) * side], 0, height) / height
            if x2 <= x1 or y2 <= y1:
                continue
            detection = SceneDetection(int(prompt), kind, label,
                                       [float(x1), float(y1), float(x2), float(y2)], float(score))
            groups["DOOR" if kind == "PASSAGE" else kind].append(detection)
        detections = []
        for candidates in groups.values():
            if not candidates:
                continue
            rectangles = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in (item.box for item in candidates)]
            keep = cv2.dnn.NMSBoxes(rectangles, [item.score for item in candidates], 0., .5)
            detections.extend(candidates[int(index)] for index in np.asarray(keep).reshape(-1))
        return sorted(detections, key=lambda item: item.score, reverse=True)
