"""InsightFace (SCRFD-детекция + ArcFace-эмбеддинги, пакет buffalo_l).

Модели хранятся локально в models/<имя_пакета>/ (скачиваются один раз).
Для кадров используется det_size 320 (кропы области головы — быстро),
для загрузки фото сотрудника — 640 (надёжнее на полных фотографиях).
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("app.ai.face")


@dataclass
class FaceInfo:
    bbox: np.ndarray          # xyxy в координатах входного изображения
    det_score: float          # уверенность детекции
    embedding: np.ndarray     # 512-d, L2-нормирован (косинус = скалярное произведение)
    aligned: np.ndarray | None  # настоящее выровненное лицо 112x112 BGR или отказ


class FaceEngine:
    """Обёртка FaceAnalysis. Один экземпляр — один det_size (потокобезопасность
    обеспечивается отдельными экземплярами для воркера и для загрузки фото)."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        root: str = ".",
        providers: Optional[list[str]] = None,
        det_size: int = 640,
        det_thresh: float = 0.5,
    ):
        from insightface.app import FaceAnalysis  # импорт тяжёлый — лениво

        self.app = FaceAnalysis(
            name=model_name,
            root=root,
            providers=providers or ["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)
        providers_used = "?"
        for _name, model in self.app.models.items():
            providers_used = model.session.get_providers()[0]
            break
        logger.info(
            "FaceEngine готова: %s, det_size=%d, провайдер=%s",
            model_name, det_size, providers_used,
        )

    def detect(self, img_bgr: np.ndarray, max_num: int = 0) -> list[FaceInfo]:
        """Все найденные лица (отсортированы по det_score по убыванию)."""
        from insightface.utils.face_align import norm_crop

        faces = self.app.get(img_bgr, max_num=max_num)
        result = []
        for f in faces:
            embedding = getattr(f, "normed_embedding", None)
            if embedding is None:
                embedding = f.embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            # insightface 2.0 не отдаёт выровненное лицо — выравниваем сами
            # по 5 ключевым точкам (шаблон ArcFace 112x112)
            aligned = None
            kps = getattr(f, "kps", None)
            if kps is not None:
                try:
                    aligned = norm_crop(img_bgr, np.asarray(kps), image_size=112)
                except Exception:
                    aligned = None
            result.append(FaceInfo(
                bbox=np.asarray(f.bbox, dtype=np.float32),
                det_score=float(f.det_score),
                embedding=np.asarray(embedding, dtype=np.float32),
                aligned=aligned,
            ))
        return result
