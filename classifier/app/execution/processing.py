import base64
import logging

import cv2
import numpy as np

from aliases import BoundingBox, CroppedBox
from execution.classification import AnimalsClassifier
from schemas.processing_info import ReceivedData, ResultingData
from utils.benchmark import timer


_logger = logging.getLogger("animals_classifier")


class ClassifierProcessor:
    def __init__(self):
        self.classifier_model = AnimalsClassifier()

    @timer
    def classify(self, data: ReceivedData) -> ResultingData:
        classes = []
        try:
            bboxes = data["bboxes"]
            image = self._b64_to_ndarray(data["img_bytes"])
            for bbox in bboxes:
                bbox = self._cvt_xywh_to_x1y1x2y2(bbox)
                image = self._crop_image(image, bbox)
                classes.append(self.classifier_model.classify_image(image))
        except Exception as ex:
            _logger.info(ex)

        return {"bboxes": bboxes, "classes": classes}

    def _crop_image(self, image: np.ndarray, bbox: CroppedBox) -> np.ndarray:
        return image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    def _b64_to_ndarray(self, image: str) -> np.ndarray:
        image = base64.b64decode(image)
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, flags=1)
        return image

    def _cvt_xywh_to_x1y1x2y2(self, bbox: BoundingBox) -> CroppedBox:
        assert (
            len(bbox) >= 4
        ), f"Bounding box must have at least 4 coordinates!. Got {len(bbox)}"
        x1, y1, w, h = bbox[:4]
        x2 = w + x1
        y2 = h + y1
        return (int(x1), int(y1), int(x2), int(y2))
