from pydantic import BaseModel

from aliases import BoundingBoxes


class ReceivedData(BaseModel):
    img_bytes: str
    bboxes: BoundingBoxes


class ResultingData(BaseModel):
    bboxes: BoundingBoxes
    classes: list[str]
