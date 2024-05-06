import logging

from aliases import BoundingBoxes
from broker.celery import celery_app
from execution.processing import ClassifierProcessor


_logger = logging.getLogger("animals_classifier")

_logger.debug("animals classifier started")
classifier = ClassifierProcessor()


@celery_app.task(name="classifier.classify_animals")
def execute_query(img_bytes: str, bboxes: BoundingBoxes):
    assert bboxes and img_bytes, "Classifier received no data"
    return classifier.classify(
        {
            "img_bytes": img_bytes,
            "bboxes": bboxes,
        }
    )
