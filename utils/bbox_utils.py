import numpy as np


def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]


def inner_bbox(bbox, shrink=0.1):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    return (
        int(x1 + shrink * w),
        int(y1 + shrink * h),
        int(x2 - shrink * w),
        int(y2 - shrink * h),
    )


def bbox_iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter)


def xyxy_to_xywh(boxes):
    """
    boxes: np.array of shape (N,4) [x1, y1, x2, y2]
    returns: np.array of shape (N,4) [x_center, y_center, width, height]
    """
    # Convert [x1, y1, x2, y2] -> [x_center, y_center, width, height]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2
    y_center = y1 + h / 2
    return np.stack([x_center, y_center, w, h], axis=1).astype(float)


def xywh_to_xyxy(boxes):
    """
    boxes: np.array of shape (N,4) with [x_center, y_center, width, height]
    returns: np.array of shape (N,4) with [x1, y1, x2, y2]
    """
    # Convert [x_center, y_center, width, height] -> [x1, y1, x2, y2]
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.stack([x1, y1, x2, y2], axis=1).astype(float)
