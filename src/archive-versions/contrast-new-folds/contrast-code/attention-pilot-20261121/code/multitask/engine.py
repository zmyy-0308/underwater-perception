"""Train, resume and evaluate one reproducible segmentation or stereo job."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import StereoSpec
from .metrics import DepthMetrics, SegMetrics, semantic_loss
from .models import Group, build, build_depth_encoder
from .stereo import StereoNet, supervised_depth_loss


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(tmp, path)


def save_model(path, payload):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def digest(path):
    with Path(path).open("rb") as stream:
        value = hashlib.sha256()
        for piece in iter(lambda: stream.read(1024*1024), b""):
            value.update(piece)
        return value.hexdigest()


def torchify_state():
    s = np.random.get_state()
    return {"python": random.getstate(), "numpy": (s[0], s[1].tolist(), *s[2:]),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_state(state):
    random.setstate(state["python"])
    s = state["numpy"]
    np.random.set_state((s[0], np.asarray(s[1], dtype=np.uint32), *s[2:]))
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def create_model(meta, weights=None):
    mode = meta["stage"]
    seed = meta["seed"]
    if mode == "depth":
        semantic = build_depth_encoder(weights, seed)
    else:
        semantic = build(Group(**meta["group"]), weights, seed)
    if mode == "seg":
        return semantic
    torch.manual_seed(seed+77)
    return StereoNet(semantic, StereoSpec(**meta["stereo"]))


def restore_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = create_model(state["meta"])
    model.load_state_dict(state["model"], strict=True)
    return model.to(device), state["meta"]


def loader(dataset, batch, seed=None):
    return DataLoader(dataset, batch_size=batch, shuffle=seed is not None, drop_last=seed is not None,
                      generator=torch.Generator().manual_seed(seed) if seed is not None else None,
                      num_workers=0)


def batches(data):
    while True:
        yield from data


def synchronized(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_training(meta, out, semantic_data, stereo_data, weights=None, parent=None, resume=False, stop_after=None):
    out, device = Path(out), torch.device(meta["device"])
    seeds = meta["seed"]
    random.seed(seeds)
    np.random.seed(seeds)
    torch.manual_seed(seeds)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seeds)
    if out.exists() and not resume:
        raise FileExistsError(f"Training output already exists: {out}")
    if resume and not (out/"last.pt").is_file():
        raise FileNotFoundError(f"No checkpoint to resume: {out/'last.pt'}")
    out.mkdir(parents=True, exist_ok=True)
    amp = device.type == "cuda"
    model = create_model(meta, weights if not resume else None)
    if parent and not resume:
        state = torch.load(parent, map_location="cpu", weights_only=True)
        source = state["meta"]
        if (source["stage"] != "seg" or source["group"] != meta["group"]
                or source["fold"] != meta["fold"] or source["seed"] != meta["seed"]
                or source["semantic_signature"] != meta["semantic_signature"]
                or state["epoch"] != source["epochs"]):
            raise ValueError("Expected matched completed semantic checkpoint for joint initialization")
        model.semantic.load_state_dict(state["model"], strict=True)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=meta["lr"], weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, meta["epochs"], eta_min=meta["min_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    begin, history = 0, []
    if resume:
        state = torch.load(out/"last.pt", map_location="cpu", weights_only=True)
        if state["meta"] != meta:
            raise ValueError("Resume requires the original architecture, data and schedule")
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, begin = state["history"], state["epoch"]
        restore_state(state["rng"])
    json_write(out/"config.json", meta)
    accum = meta["effective_batch"]//meta["batch"]
    end = meta["epochs"] if stop_after is None else min(meta["epochs"], stop_after)
    for i in range(begin, end):
        t0 = perf_counter()
        model.train()
        if amp:
            torch.cuda.reset_peak_memory_stats(device)
        if semantic_data is not None:
            semantic_data.epoch = i
        seg = loader(semantic_data, meta["batch"], seeds+20000+i) if semantic_data is not None else None
        depth = loader(stereo_data, meta["batch"], seeds+40000+i) if stereo_data is not None else None
        if any(len(dataset) == 0 for dataset in (seg, depth) if dataset is not None):
            raise ValueError("Each training source must supply at least one physical batch")
        total = max(len(seg) if seg is not None else 0, len(depth) if depth is not None else 0)
        total = meta["steps"] or total
        seg_batches = batches(seg) if seg is not None else None
        depth_batches = batches(depth) if depth is not None else None
        totals = {"seg": 0., "depth": 0.}
        opt.zero_grad(set_to_none=True)
        for k in range(total):
            block = min(accum, total-(k//accum)*accum)
            tasks = []
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                if seg_batches is not None:
                    x, target, _ = next(seg_batches)
                    x, target = x.to(device), target.to(device)
                    logits = model(x) if meta["stage"] == "seg" else model.segment(x)
                    tasks.append(("seg", semantic_loss(logits, target), 1.))
                if depth_batches is not None:
                    l, r, z, valid, fb, _ = next(depth_batches)
                    z, valid = z.to(device), valid.to(device)
                    pred = model(l.to(device), r.to(device), fb.to(device))
                    tasks.append(("depth", supervised_depth_loss(pred, z, valid), meta["depth_weight"]))
                loss = sum(value*weight for _, value, weight in tasks)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            scaler.scale(loss/block).backward()
            for name, value, _ in tasks:
                totals[name] += float(value.detach())
            if (k+1) % accum == 0 or k+1 == total:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        synchronized(device)
        history.append({"epoch": i+1, "seg_loss": totals["seg"]/total if seg is not None else None,
                        "depth_loss": totals["depth"]/total if depth is not None else None,
                        "lr": opt.param_groups[0]["lr"], "seconds": perf_counter()-t0,
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if amp else None})
        schedule.step()
        save_model(out/"last.pt", {"meta": meta, "epoch": i+1, "model": model.state_dict(),
                                   "optimizer": opt.state_dict(), "scheduler": schedule.state_dict(),
                                   "scaler": scaler.state_dict(), "rng": torchify_state(), "history": history})
        json_write(out/"history.json", history)
        print(history[-1], flush=True)
    if max(begin, end) == meta["epochs"] and not (out/"model.pt").is_file():
        state = torch.load(out/"last.pt", map_location="cpu", weights_only=True)
        save_model(out/"model.pt", {"meta": meta, "epoch": state["epoch"], "model": state["model"]})
    return history


@torch.inference_mode()
def evaluate(model, meta, sem, stereo, device):
    model.eval()
    payload = {}
    rows = []
    if sem is not None:
        metric = SegMetrics(meta["group"]["task"])
        forward = model if meta["stage"] == "seg" else model.segment
        for x, target, names in loader(sem, 1):
            stats = metric.update(forward(x.to(device)), target)
            rows.extend({"task": "sem", "id": name, **row} for name, row in zip(names, stats))
        payload["semantic"] = {**metric.result(), "images": len(sem), "signature": sem.signature}
    if stereo is not None:
        metric = DepthMetrics()
        for l, r, z, valid, fb, names in loader(stereo, 1):
            stats = metric.update(model(l.to(device), r.to(device), fb.to(device)).cpu(), z, valid)
            rows.extend({"task": "depth", "id": name, **row} for name, row in zip(names, stats))
        payload["depth"] = {**metric.result(), "images": len(stereo), "signature": stereo.signature}
    return payload, rows
