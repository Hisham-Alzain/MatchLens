import numpy as np
import cv2


def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]


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


def inner_bbox(bbox, shrink=0.15):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    return (
        int(x1 + shrink * w),
        int(y1 + shrink * h),
        int(x2 - shrink * w),
        int(y2 - shrink * h),
    )


def crop_jersey(image, bbox, top=0.15, bottom=0.55, left=0.15, right=0.85):
    """
    Crops a tighter t-shirt region from a player bounding box.
    """
    # 1. Unpack bbox. Detections often come as floats, so we cast them to int immediately.
    x1, y1, x2, y2 = bbox

    # Ensure base coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    w = x2 - x1
    h = y2 - y1

    # 2. Calculate new coordinates and force int casting on the RESULT
    x1_new = int(x1 + (w * left))
    x2_new = int(x1 + (w * right))
    y1_new = int(y1 + (h * top))
    y2_new = int(y1 + (h * bottom))

    # 3. Safety check: Ensure we don't slice outside the image dimensions
    # (Optional but recommended to prevent empty crops or errors at edges)
    h_img, w_img = image.shape[:2]
    x1_new = max(0, x1_new)
    y1_new = max(0, y1_new)
    x2_new = min(w_img, x2_new)
    y2_new = min(h_img, y2_new)

    return image[y1_new:y2_new, x1_new:x2_new]


def grabcut_player(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[int(y1) : int(y2), int(x1) : int(x2)]

    mask = np.zeros(crop.shape[:2], np.uint8)
    rect = (0, 0, crop.shape[1] - 1, crop.shape[0] - 1)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(crop, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Keep only probable foreground and foreground
    player_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

    return player_mask, crop
