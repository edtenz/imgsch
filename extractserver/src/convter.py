from detect import BoundingBox


def detect_boxes(bboxes: list[BoundingBox]) -> list[dict]:
    return [box.to_dict() for box in bboxes]
