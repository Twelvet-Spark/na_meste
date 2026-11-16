"""Bundle-backed OSNet-AIN x1.0 body appearance embeddings, not face identity.

The approved dynamic-batch ONNX export consumes RGB 256x128 images with
ImageNet normalization. PIL bilinear resizing matches the model preparer's
reference path. Every real crop is embedded in order, in chunks of at most 16;
body appearance alone never establishes an employee identity.
"""
import hashlib
import json
import logging

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.ai.model_bundle import ModelBundle, ModelBundleError
from app.ai.providers import model_providers

logger = logging.getLogger("app.ai.reid")

INPUT_H, INPUT_W = 256, 128
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PREPROCESSING_ID = "osnet-rgb256x128-imagenet-v1"
_MAX_BATCH = 16
EMBEDDING_DIM = 512


def _invalid(detail: str) -> ModelBundleError:
    return ModelBundleError("MODEL_CONTRACT_INVALID", "body", detail)


def _dynamic_float_tensor(value, dimensions: list[int]) -> bool:
    shape = value.shape
    return (
        value.type == "tensor(float)"
        and isinstance(shape, list)
        and len(shape) == len(dimensions) + 1
        and isinstance(shape[0], str)
        and bool(shape[0])
        and shape[1:] == dimensions
    )


class PersonReID:
    def __init__(self, bundle: ModelBundle, providers: list):
        model_path = bundle.require("body")
        declaration = bundle.metadata["components"]["body"]
        if declaration["preprocessing_id"] != _PREPROCESSING_ID:
            raise _invalid("Unsupported body preprocessing")
        identity = {
            "preprocessing_id": declaration["preprocessing_id"],
            "files": {file["role"]: file["sha256"] for file in declaration["files"]},
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        self.model_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        try:
            self.session = ort.InferenceSession(
                str(model_path), providers=model_providers(providers)
            )
        except Exception as exc:
            raise _invalid("Body model session cannot be initialized") from exc
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or not _dynamic_float_tensor(inputs[0], [3, INPUT_H, INPUT_W])
            or not _dynamic_float_tensor(outputs[0], [EMBEDDING_DIM])
        ):
            raise _invalid("Expected dynamic float32 NCHW input and N-by-512 output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        logger.info(
            "Re-ID модель загружена: %s (провайдеры=%s)",
            self.model_id, self.session.get_providers(),
        )

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return all N normalized float32 vectors for uint8 HWC BGR crops.

        Invalid images, malformed model outputs and native inference errors fail
        closed with the static body MODEL_CONTRACT_INVALID error.
        """
        if not crops:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        result = np.empty((len(crops), EMBEDDING_DIM), dtype=np.float32)
        for start in range(0, len(crops), _MAX_BATCH):
            stop = min(start + _MAX_BATCH, len(crops))
            batch = np.empty((stop - start, 3, INPUT_H, INPUT_W), dtype=np.float32)
            for row, index in enumerate(range(start, stop)):
                crop = crops[index]
                if (
                    not isinstance(crop, np.ndarray)
                    or crop.dtype != np.uint8
                    or crop.ndim != 3
                    or crop.shape[2] != 3
                    or crop.shape[0] == 0
                    or crop.shape[1] == 0
                ):
                    raise _invalid("Body crop must be a nonempty uint8 HWC BGR image")
                image = Image.fromarray(crop[:, :, ::-1]).resize(
                    (INPUT_W, INPUT_H), Image.Resampling.BILINEAR
                )
                value = np.asarray(image, dtype=np.float32)
                value /= 255.0
                value -= _MEAN
                value /= _STD
                batch[row] = value.transpose(2, 0, 1)
            try:
                output = self.session.run([self.output_name], {self.input_name: batch})[0]
            except Exception as exc:
                raise _invalid("Body inference failed") from exc
            if (
                not isinstance(output, np.ndarray)
                or output.dtype != np.float32
                or output.shape != (stop - start, EMBEDDING_DIM)
                or not np.isfinite(output).all()
            ):
                raise _invalid("Body output must contain one finite float32 512-vector per crop")
            # Accumulate in float64 without copying the full output; finite FP32
            # vectors can otherwise overflow/underflow during norm computation.
            norms = np.einsum("ij,ij->i", output, output, dtype=np.float64)
            np.sqrt(norms, out=norms)
            if not np.isfinite(norms).all() or np.any(norms <= 0):
                raise _invalid("Body output has an invalid L2 norm")
            np.divide(output, norms[:, None], out=result[start:stop])
        return result
