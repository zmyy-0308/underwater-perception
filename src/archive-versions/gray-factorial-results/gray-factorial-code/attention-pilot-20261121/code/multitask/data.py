"""Manifest-backed semantic masks and FLSea Stereo members read directly from ZIP."""
import csv
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
import torch
from .canny import rgb_with_canny
from torch.utils.data import Dataset
import yaml

from .models import SIZES

SEQUENCES = ("canyon1", "canyon2", "rock_garden1", "rock_garden2")


def rgb_tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()/255


def picture(path, kind):
    with Image.open(path) as im:
        if kind == "rgb":
            im = im.convert("RGB")
        return np.array(im)


def select_even(rows, limit):
    if limit and len(rows) > limit:
        return [rows[i] for i in np.linspace(0, len(rows)-1, limit, dtype=int)]
    return rows


def pipe_folds(rows, fold):
    if not 0 <= fold < 5:
        raise ValueError("Pipe fold must be in [0,4]")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["sequence"]].append((float(row["timestamp"]), i))
    fit, test = [], []
    for entries in groups.values():
        entries.sort()
        ts = np.array([t for t, _ in entries])
        if not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
            raise ValueError("Pipeline timestamps must be finite and unique within a sequence")
        block = np.array_split(np.arange(len(entries)), 5)[fold]
        if not len(block):
            fit.extend(i for _, i in entries)
            continue
        lo, hi = ts[block[0]], ts[block[-1]]
        test.extend(entries[j][1] for j in block)
        fit.extend(i for t, i in entries if t < lo-2 or t > hi+2)
    return {"train": sorted(fit), "test": sorted(test)}


class SemanticData(Dataset):
    def __init__(self, task, path, split, fold=0, limit=0, augment=False, seed=42):
        if task not in SIZES:
            raise ValueError(f"Unknown semantic task {task}")
        self.task, self.size, self.augment, self.seed = task, SIZES[task], augment, seed
        self.epoch = 0
        self.arrays = None
        self.canny = False
        path = Path(path).resolve()
        if path.is_dir():
            if task != "bio":
                raise ValueError("A directory input is supported only for the SUIM NPY cache")
            meta = json.loads((path/"meta.json").read_text(encoding="utf-8-sig"))
            split_meta = meta["splits"].get(split)
            if not split_meta:
                raise ValueError(f"SUIM cache has no {split!r} split")
            images = np.load(path/f"{split}_img.npy", mmap_mode="r")
            masks = np.load(path/f"{split}_msk.npy", mmap_mode="r")
            if not len(images) == len(masks) == len(split_meta["stems"]):
                raise ValueError("SUIM cache index/array lengths differ")
            self.arrays = (images, masks)
            self.rows = [{"id": stem, "index": i, "encoding": "index"}
                         for i, stem in enumerate(split_meta["stems"])]
            paths = [path/n for n in ("meta.json", f"{split}_img.npy", f"{split}_msk.npy")]
            source = "\n".join(f"{p}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in paths)
        else:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            ids = [r["id"] for r in rows]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate semantic image IDs")
            for row in rows:
                for field in ("image", "mask"):
                    row[field] = (path.parent/row[field]).resolve()
            if task == "pipe" and all(not r.get("split") for r in rows):
                by_split = pipe_folds(rows, fold)
                self.rows = [rows[i] for i in by_split.get(split, [])]
            else:
                if task == "bio":
                    sequences = defaultdict(set)
                    for row in rows:
                        identity = row.get("group") or row.get("sequence")
                        if identity:
                            sequences[identity].add(row["split"])
                    if any(len(value) > 1 for value in sequences.values()):
                        raise ValueError("SUIM sequence or duplicate group spans splits")
                self.rows = [r for r in rows if r.get("split") == split]
                if task == "pipe" and "fold" in (self.rows[0] if self.rows else {}):
                    self.rows = [r for r in self.rows if int(r["fold"]) == fold]
            manifest_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            paths = [r[k] for r in self.rows for k in ("image", "mask")]
            source = manifest_digest+"\n"+"\n".join(f"{p}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in paths)
        self.rows = select_even(self.rows, limit)
        if not self.rows:
            raise ValueError(f"No semantic samples for {task}/{split}/fold {fold}")
        self.signature = hashlib.sha256((source+str((task, split, fold, limit))).encode()).hexdigest()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.arrays:
            im, mask = (np.asarray(a[row["index"]]) for a in self.arrays)
        else:
            im = picture(row["image"], "rgb")
            encoding = row.get("encoding") or ("binary" if self.task == "pipe" else "index")
            mask = picture(row["mask"], "rgb" if encoding == "rgb" else "raw")
            if encoding == "rgb":
                mask = ((mask[..., 0] > 127)*4+(mask[..., 1] > 127)*2+(mask[..., 2] > 127)).astype(np.uint8)
            elif encoding == "binary":
                if mask.ndim == 3:
                    mask = mask[..., :3].max(axis=-1)
                mask = (mask > int(row.get("threshold") or 0)).astype(np.uint8)
            elif encoding != "index":
                raise ValueError(f"Unknown mask encoding {encoding}")
        if mask.ndim != 2 or im.shape[:2] != mask.shape:
            raise ValueError(f"Mask and image must be aligned: {row['id']}")
        classes = 2 if self.task == "pipe" else 8
        if np.any((mask >= classes) & (mask != 255)) or mask.min() < 0:
            raise ValueError(f"Invalid semantic label: {row['id']}")
        h, w = self.size
        if self.task == "bio":
            # Preserve the historical short-side resize and square crop.
            scale = h/min(im.shape[:2])
            nw, nh = round(im.shape[1]*scale), round(im.shape[0]*scale)
        else:
            nw, nh = w, h
        if im.shape[:2] != (nh, nw):
            im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        x = rng.integers(nw-w+1) if self.augment else (nw-w)//2
        y = rng.integers(nh-h+1) if self.augment else (nh-h)//2
        im, mask = im[y:y+h, x:x+w], mask[y:y+h, x:x+w]
        if self.augment:
            if rng.random() < .5:
                im, mask = im[:, ::-1], mask[:, ::-1]
            im = np.clip(im.astype(np.float32)*rng.uniform(.85, 1.15)+rng.uniform(-12, 12), 0, 255).astype(np.uint8)
        pixels = rgb_tensor(im)
        if self.canny:
            pixels = rgb_with_canny(pixels)
        return pixels, torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64)), row["id"]


def stereo_inventory(archive):
    entries = {sequence: {} for sequence in SEQUENCES}
    pattern = r"(canyon[12]|rock_garden[12])/\1/(imgs|depth)/(LFT|RGT)/(?:LFT|RGT)_(.+?)(?:_abs_depth)?\.tif"
    for member in archive.infolist():
        match = re.fullmatch(pattern, member.filename)
        if match:
            seq, kind, side, frame = match.groups()
            fields = entries[seq].setdefault(frame, {})
            k = f"{kind}_{side}"
            if k in fields:
                raise ValueError(f"Duplicate FLSea member {member.filename}")
            fields[k] = member
    for seq, frames in entries.items():
        if not frames or any(len(f) != 4 for f in frames.values()):
            raise ValueError(f"Missing FLSea image/depth pairs in {seq}")
    return entries


@dataclass(frozen=True)
class StereoSpec:
    size: tuple = (288, 512)
    direction: str = "stored-ltr"
    depth_kind: str = "z"
    min_depth: float = .5
    max_depth: float = 15.
    shifts: int = 64

    def __post_init__(self):
        if (len(self.size) != 2 or min(self.size) < 32 or any(s % 8 for s in self.size)
                or self.direction not in ("stored-ltr", "stored-rtl")
                or self.depth_kind not in ("z", "range")
                or not 0 < self.min_depth < self.max_depth or not 2 <= self.shifts <= self.size[1]//8):
            raise ValueError(f"Invalid stereo specification: {self}")


class Rectification:
    def __init__(self, source, spec):
        lines = source.decode("utf-8-sig").splitlines()
        cfg = yaml.safe_load("\n".join(line for line in lines if not line.startswith("%YAML"))
                             .replace("!!opencv-matrix", ""))
        width, height = cfg["camera.width"], cfg["camera.height"]
        self.source_shape = (height, width)
        shape = (width, height)
        out_h, out_w = spec.size
        intr, dist = [], []
        for side in ("lft", "rgt"):
            fx, fy, cx, cy = (cfg[f"camera_{side}.{v}"] for v in ("fx", "fy", "cx", "cy"))
            intr.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float))
            dist.append(np.array([cfg[f"camera_{side}.{v}"] for v in ("k1", "k2", "p1", "p2")]))
        transform = np.asarray(cfg["Stereo.T_c1_c2"]["data"]).reshape(4, 4)
        if spec.direction == "stored-rtl":
            transform = np.linalg.inv(transform)
        translation = (transform[:3, 3]/1000).reshape(3, 1)
        r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(intr[0], dist[0], intr[1], dist[1], shape,
            transform[:3, :3], translation, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
            newImageSize=(out_w, out_h))
        self.fb = float(-p2[0, 3])
        if self.fb <= 0:
            raise ValueError("Calibration direction does not produce positive horizontal disparity")
        self.maps = [cv2.initUndistortRectifyMap(k, d, r, p, (out_w, out_h), cv2.CV_32FC1)
                     for k, d, r, p in zip(intr, dist, (r1, r2), (p1, p2))]
        mx, my = self.maps[0]
        uv = np.stack((np.rint(mx), np.rint(my)), axis=-1).astype(np.float32)
        rays_xy = cv2.undistortPoints(uv.reshape(-1, 1, 2), intr[0], dist[0]).reshape(out_h, out_w, 2)
        rays = np.concatenate((rays_xy, np.ones((out_h, out_w, 1))), axis=-1)
        correction = rays@r1[2]
        if spec.depth_kind == "range":
            correction /= np.linalg.norm(rays, axis=-1)
        self.z_scale = correction.astype(np.float32)
        self.pixel_x = np.arange(out_w)[None, :]
        self.spec = spec

    def convert(self, left, right, depth):
        if left.shape != (*self.source_shape, 3) or right.shape != left.shape or depth.shape != self.source_shape:
            raise ValueError("FLSea raster shape does not match stereo calibration")
        rectified = [cv2.remap(im, *maps, cv2.INTER_LINEAR) for im, maps in zip((left, right), self.maps)]
        z = (cv2.remap(depth, *self.maps[0], cv2.INTER_NEAREST)*self.z_scale).astype(np.float32)
        valid = np.isfinite(z) & (z >= self.spec.min_depth) & (z <= self.spec.max_depth)
        disparity = self.fb/np.where(valid, z, 1)
        x_right = np.rint(self.pixel_x-disparity).astype(int)
        right_x, right_y = self.maps[1]
        safe_x = x_right.clip(0, right_x.shape[1]-1)
        target_x = np.take_along_axis(right_x, safe_x, axis=1)
        target_y = np.take_along_axis(right_y, safe_x, axis=1)
        h, w = self.source_shape
        valid &= (x_right >= 0) & (disparity <= 8*(self.spec.shifts-1))
        valid &= (target_x >= 0) & (target_x < w-1) & (target_y >= 0) & (target_y < h-1)
        return (*map(rgb_tensor, rectified), torch.from_numpy(np.where(valid, z, 0)),
                torch.from_numpy(valid), torch.tensor(self.fb, dtype=torch.float32))


class StereoData(Dataset):
    def __init__(self, path, sequences, spec, limit=0):
        self.path = Path(path).resolve()
        self._zip = None
        self._owner = None
        if not sequences or set(sequences)-set(SEQUENCES) or len(set(sequences)) != len(sequences):
            raise ValueError("Provide unique FLSea sequence names")
        with ZipFile(self.path) as source:
            entries = stereo_inventory(source)
            calibration = source.read("calibration/calibration/dive1_2025_recalc.yaml")
            self.rectifier = Rectification(calibration, spec)
            self.counts = {key: len(rows) for key, rows in entries.items()}
            self.rows = []
            sig = hashlib.sha256(calibration+str(spec).encode())
            for seq in sequences:
                for frame, files in select_even(sorted(entries[seq].items()), limit):
                    for member in files.values():
                        sig.update(f"{member.filename}:{member.CRC}:{member.file_size}".encode())
                    self.rows.append((f"{seq}/{frame}", {k: v.filename for k, v in files.items()}))
            self.signature = sig.hexdigest()

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        return {**self.__dict__, "_zip": None, "_owner": None}

    def raw(self, index):
        if self._owner != os.getpid() or self._zip is None:
            if self._zip is not None:
                self._zip.close()
            self._zip, self._owner = ZipFile(self.path), os.getpid()
        name, members = self.rows[index]
        images = [picture(BytesIO(self._zip.read(members[k])), "raw")
                  for k in ("imgs_LFT", "imgs_RGT", "depth_LFT")]
        return (*images, name)

    def __getitem__(self, index):
        left, right, depth, name = self.raw(index)
        return (*self.rectifier.convert(left, right, depth), name)
