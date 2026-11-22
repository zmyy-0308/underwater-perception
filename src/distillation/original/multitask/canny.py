"""Fixed Canny input and contour-only reference for raw RGB frames."""

import cv2
import numpy as np
import torch
from torch.nn import functional as F


CANNY_MIN_HIGH = 12
CANNY_HIGH_QUANTILE = 0.90
CANNY_LOW_RATIO = 0.40
CANNY_GROUPS = frozenset(("C1", "C3"))
CONTRAST_GROUPS = frozenset(("C2", "C3", "C5", "C6"))
CONTRAST_GROUPS_ALL = CANNY_GROUPS | CONTRAST_GROUPS


def local_gray_difference(rgb):
    y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    horizontal = F.avg_pool2d(y, (1, 9), stride=1, padding=(0, 4), count_include_pad=False)
    local_mean = F.avg_pool2d(horizontal, (9, 1), stride=1, padding=(4, 0), count_include_pad=False)
    return y - local_mean


def canny_edges(image):
    gray = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    gradient = cv2.add(cv2.convertScaleAbs(dx), cv2.convertScaleAbs(dy))
    histogram = cv2.calcHist([gradient], [0], None, [256], [0, 256]).ravel()
    positive = histogram[1:].sum()
    if positive == 0:
        return np.zeros_like(gray)
    high = max(CANNY_MIN_HIGH, int(np.searchsorted(
        np.cumsum(histogram), histogram[0] + CANNY_HIGH_QUANTILE * positive
    )))
    return cv2.Canny(dx, dy, high * CANNY_LOW_RATIO, high, L2gradient=False)


def rgb_with_canny(rgb):
    image = rgb.mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    edge = torch.from_numpy(canny_edges(image)).to(rgb.dtype).div_(255)
    return torch.cat((rgb, edge.unsqueeze(0)), dim=0)


def canny_contour_mask(image):
    edges = canny_edges(image)
    connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = edges.size
    eligible = []
    for contour in contours:
        coverage = cv2.contourArea(contour)
        if not 0.005 * area <= coverage <= 0.50 * area:
            continue
        _, (width, height), _ = cv2.minAreaRect(contour)
        if min(width, height) and max(width, height) / min(width, height) >= 2:
            eligible.append((coverage, contour))
    mask = np.zeros_like(edges)
    if eligible:
        _, contour = max(eligible, key=lambda item: item[0])
        cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED)
    return mask
