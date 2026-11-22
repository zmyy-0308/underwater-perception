"""Dataset-level semantic counters and validity-aware metric depth errors."""
import cv2
import numpy as np
import torch
from torch.nn import functional as F

BIO_CLASSES = ("BW", "HD", "PF", "WR", "RO", "RI", "FV", "SR")


def ratio(a, b):
    return float(a/b) if b else None


def semantic_loss(logits, target):
    if logits.ndim != 4 or logits.shape[0] != len(target) or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError("Semantic output shape differs from mask shape")
    valid = target != 255
    if not valid.any():
        raise ValueError("Semantic batch has no supervised pixels")
    c = logits.shape[1]
    targets = F.one_hot(target.masked_fill(~valid, 0), c).permute(0, 3, 1, 2)[:, 1:].float()
    prob = logits.float().softmax(1)[:, 1:]*valid[:, None]
    targets *= valid[:, None]
    axes = (0, 2, 3)
    dice = (2*(prob*targets).sum(axes)+1)/(prob.sum(axes)+targets.sum(axes)+1)
    return F.cross_entropy(logits, target, ignore_index=255)+(1-dice).mean()


def boundary_hits(predicted, target, valid, tolerance):
    kernel = np.ones((3, 3), dtype=np.uint8)
    safe = cv2.erode(valid.astype(np.uint8), kernel) > 0

    def edge(mask):
        im = mask.astype(np.uint8)
        return (mask & ~(cv2.erode(im, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0)) & safe

    pred, truth = edge(predicted), edge(target)
    count_p, count_t = int(pred.sum()), int(truth.sum())
    if not (count_p and count_t):
        return np.array([0, count_p, 0, count_t], dtype=np.int64)
    distance_t = cv2.distanceTransform((~truth).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance_p = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.array([int((pred & (distance_t <= tolerance)).sum()), count_p,
                     int((truth & (distance_p <= tolerance)).sum()), count_t], dtype=np.int64)


class SegMetrics:
    def __init__(self, task, boundary_px=2):
        self.task = task
        self.names = ("background", "pipe") if task == "pipe" else BIO_CLASSES
        self.matrix = np.zeros((len(self.names), len(self.names)), dtype=np.int64)
        self.boundary = np.zeros(4, dtype=np.int64)
        self.boundary_px = boundary_px
        self.no_fish = 0
        self.false_frames = 0

    def update(self, logits, target):
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits in segmentation evaluation")
        pred = logits.argmax(1).cpu().numpy()
        target = target.cpu().numpy()
        fish = logits.float().softmax(dim=1)[:, 6].cpu().numpy() if self.task == "bio" else None
        rows = []
        for i, (p, t) in enumerate(zip(pred, target)):
            c = len(self.names)
            valid = t != 255
            conf = np.bincount(c*t[valid]+p[valid], minlength=c*c).reshape(c, c)
            self.matrix += conf
            row = {"confusion": conf.tolist(), "valid_px": int(valid.sum())}
            if self.task == "pipe":
                boundary = boundary_hits(p == 1, t == 1, valid, self.boundary_px)
                self.boundary += boundary
                row["boundary_counts"] = boundary.tolist()
            if self.task == "bio" and valid.any() and not (t == 6).any():
                self.no_fish += 1
                _, _, stats, _ = cv2.connectedComponentsWithStats(((fish[i] > .5) & valid).astype(np.uint8),
                                                                  connectivity=8)
                false = bool((stats[1:, cv2.CC_STAT_AREA] >= 400).any())
                self.false_frames += false
                row["no_fish_false"] = false
            rows.append(row)
        return rows

    def result(self):
        out = {}
        for i, name in enumerate(self.names):
            tp, truth, predicted = self.matrix[i, i], self.matrix[i].sum(), self.matrix[:, i].sum()
            out[name] = {"iou": ratio(tp, truth+predicted-tp), "dice": ratio(2*tp, truth+predicted),
                         "precision": ratio(tp, predicted), "recall": ratio(tp, truth)}
        ious = [v["iou"] for v in out.values() if v["iou"] is not None]
        main = "pipe" if self.task == "pipe" else "FV"
        result = {"per_class": out, "primary_iou": out[main]["iou"], "miou": float(np.mean(ious)) if ious else None}
        if self.task == "pipe":
            a, b, c, d = self.boundary
            precision, recall = ratio(a, b) or 0., ratio(c, d) or 0.
            result["boundary_f1"] = ratio(2*precision*recall, precision+recall) or 0. if b or d else None
        else:
            result.update(no_fish_frames=self.no_fish, no_fish_false_frames=self.false_frames,
                          no_fish_false_rate=ratio(self.false_frames, self.no_fish))
        return result


class DepthMetrics:
    def __init__(self):
        self.total = np.zeros(5, dtype=np.float64)
        self.raster = 0

    def update(self, prediction, truth, valid):
        rows = []
        for p, z, v in zip(prediction, truth, valid):
            good = v & torch.isfinite(p) & (p > 0)
            measured, estimate = z[good].double(), p[good].double()
            error = (estimate-measured).abs()
            counts = [int(v.sum()), int(good.sum()), float(error.sum()), float(error.square().sum()),
                      float((error/measured).sum())]
            self.total += counts
            self.raster += v.numel()
            rows.append({"gt_valid_px": counts[0], "pred_valid_px": counts[1], "ae_sum_m": counts[2],
                         "se_sum_m2": counts[3], "relative_sum": counts[4], "raster_px": v.numel()})
        return rows

    def result(self):
        gt, n, ae, se, relative = self.total
        return {"mae_m": ratio(ae, n), "rmse_m": float(np.sqrt(se/n)) if n else None,
                "abs_rel": ratio(relative, n), "gt_valid_px": int(gt), "pred_valid_px": int(n),
                "valid_coverage": ratio(n, gt), "gt_raster_coverage": ratio(gt, self.raster)}
