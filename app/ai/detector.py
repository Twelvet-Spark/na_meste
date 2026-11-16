"""Bundle-backed D-FINE-S: RGB black letterboxing and native-coordinate people."""
import logging

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.ai.model_bundle import ModelBundle, ModelBundleError
from app.ai.providers import model_providers

logger = logging.getLogger("app.ai.detector")
_SIZE = 640


class PersonDetector:
    def __init__(self, bundle: ModelBundle, providers: list, confidence: float = .1):
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Person confidence must be finite and in 0..1")
        self.confidence = confidence
        self.session = ort.InferenceSession(
            str(bundle.require("person")), providers=model_providers(providers))
        if ({value.name for value in self.session.get_inputs()} != {"images", "orig_target_sizes"}
                or {value.name for value in self.session.get_outputs()} != {"labels", "boxes", "scores"}):
            raise ModelBundleError("MODEL_CONTRACT_INVALID", "person", "D-FINE IO names")
        logger.info("D-FINE ready: %s", self.session.get_providers())

    def detect(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[tuple[np.ndarray, float]]]:
        if not frames:
            return []
        images = np.zeros((len(frames), 3, _SIZE, _SIZE), dtype=np.float32)
        transforms = []
        for index, frame in enumerate(frames):
            if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                    or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0):
                raise ValueError("Expected a nonempty BGR image")
            height, width = frame.shape[:2]
            scale = min(_SIZE / width, _SIZE / height)
            resized_width, resized_height = max(1, int(width * scale)), max(1, int(height * scale))
            pad_x, pad_y = (_SIZE - resized_width) // 2, (_SIZE - resized_height) // 2
            # Match the official PIL bilinear resize, including its downsampling
            # filter; OpenCV INTER_LINEAR is not the same image transformation.
            resized = Image.fromarray(frame[:, :, ::-1]).resize(
                (resized_width, resized_height), Image.Resampling.BILINEAR)
            region = images[index, :, pad_y:pad_y + resized_height, pad_x:pad_x + resized_width]
            region[:] = np.asarray(resized).transpose(2, 0, 1)
            region /= 255.
            transforms.append((scale, pad_x, pad_y, width, height))
        labels, boxes, scores = self.session.run(
            ["labels", "boxes", "scores"],
            {"images": images,
             "orig_target_sizes": np.full((len(frames), 2), _SIZE, dtype=np.int64)})
        if (labels.ndim != 2 or labels.shape[0] != len(frames)
                or labels.dtype.kind not in "iu"
                or boxes.shape != (*labels.shape, 4) or boxes.dtype.kind != "f"
                or scores.shape != labels.shape or scores.dtype.kind != "f"):
            raise ModelBundleError("MODEL_CONTRACT_INVALID", "person", "D-FINE batch shape/dtype")
        outcomes = []
        for index, (scale, pad_x, pad_y, width, height) in enumerate(transforms):
            accepted = ((labels[index] == 0) & np.isfinite(scores[index])
                        & (scores[index] >= self.confidence) & (scores[index] <= 1)
                        & np.isfinite(boxes[index]).all(axis=1))
            people = []
            for box, score in zip(boxes[index, accepted], scores[index, accepted]):
                native = np.array([(box[0] - pad_x) / scale, (box[1] - pad_y) / scale,
                                   (box[2] - pad_x) / scale, (box[3] - pad_y) / scale],
                                  dtype=np.float32)
                np.clip(native[::2], 0, width, out=native[::2])
                np.clip(native[1::2], 0, height, out=native[1::2])
                if native[2] > native[0] and native[3] > native[1]:
                    people.append((native, float(score)))
            # D-FINE is end-to-end: no YOLO decoder or additional NMS.
            outcomes.append(people)
        return outcomes
