"""Non-Maximum Suppression (NMS) — shared across detection stages.

Any detection stage (face/SCRFD, object/YOLO, etc.) can import this
instead of re-implementing NMS locally.
"""

from __future__ import annotations

import numpy as np


def greedy_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Standard greedy Non-Maximum Suppression.

    Parameters
    ----------
    boxes : ndarray, shape (N, 4)
        Bounding boxes as ``[x1, y1, x2, y2]``.
    scores : ndarray, shape (N,)
        Confidence scores.
    iou_threshold : float
        IoU threshold — boxes with IoU > threshold are suppressed.

    Returns
    -------
    list[int]
        Indices of boxes to keep, sorted by descending score.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
