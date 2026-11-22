"""Profile one or more frozen branches using a fixed real RGB/stereo sample."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multitask.data import StereoData, StereoSpec, picture, rgb_tensor
from multitask.canny import CANNY_GROUPS, rgb_with_canny
from multitask.engine import json_write, restore_model, synchronized
from multitask.models import SIZES


def semantic_geometry(image, task, canny=False):
    h, w = SIZES[task]
    if task == "bio":
        scale = h/min(image.shape[:2])
        dw, dh = round(image.shape[1]*scale), round(image.shape[0]*scale)
        data = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_LINEAR)
        x, y = (dw-w)//2, (dh-h)//2
        return rgb_tensor(data[y:y+h, x:x+w])[None], (x, y, dw, dh)
    tensor = rgb_tensor(cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR))
    if canny:
        tensor = rgb_with_canny(tensor)
    return tensor[None], (0, 0, w, h)


def model_cost(models, run):
    total = 0

    def hook(layer, values, output):
        nonlocal total
        if isinstance(layer, nn.Conv2d):
            total += output.numel()*layer.weight[0].numel()
        elif isinstance(layer, nn.ConvTranspose2d):
            total += values[0].numel()*layer.weight[0].numel()
        else:
            total += output.numel()*layer.in_features
    hooks = [layer.register_forward_hook(hook) for model in models for layer in model.modules()
             if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))]
    try:
        run()
    finally:
        for h in hooks:
            h.remove()
    return total


@torch.inference_mode()
def profile(args):
    device = torch.device(args.device)
    if args.fp16 and device.type != "cuda":
        raise ValueError("FP16 AMP timing uses a CUDA device")
    if args.runs < 1 or args.frames < 1 or args.warmup < 0 or args.threads < 1:
        raise ValueError("Invalid timing budget")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    entries = [restore_model(p, device) for p in args.checkpoints]
    models = [model.eval() for model, _ in entries]
    stereo = [(model, meta) for model, meta in entries if meta["stage"] != "seg"]
    semantic = [(model, meta) for model, meta in entries if meta["stage"] != "depth"]
    if len(stereo) > 1:
        raise ValueError("Time one stereo output at a time with any accompanying semantic models")
    if stereo:
        if not args.stereo_zip:
            raise ValueError("Depth profiling requires a local FLSea archive")
        config = stereo[0][1]
        sample = StereoData(args.stereo_zip, config["stereo_splits"]["test"][:1],
                            StereoSpec(**config["stereo"]), 1)
        image, right, _, frame = sample.raw(0)
    else:
        if not args.image:
            raise ValueError("Semantic profiling requires an RGB image")
        image = picture(args.image, "rgb")
        frame = str(Path(args.image).resolve())

    def inputs():
        depth = []
        left_image = image
        if stereo:
            rect = [cv2.remap(im, *maps, cv2.INTER_LINEAR) for im, maps in zip((image, right), sample.rectifier.maps)]
            depth = [rgb_tensor(im)[None].to(device) for im in rect]
            depth.append(torch.tensor([sample.rectifier.fb], device=device))
            left_image = rect[0]
        left_tensors = [
            semantic_geometry(left_image, meta["group"]["task"], meta["group"]["name"] in CANNY_GROUPS)[0].to(device)
            for _, meta in semantic
        ]
        return left_tensors, depth

    def infer(data, finish=False):
        frames, depth = data
        output = []
        with torch.autocast(device.type, dtype=torch.float16, enabled=args.fp16):
            for (model, meta), frame in zip(semantic, frames):
                logits = model(frame) if meta["stage"] == "seg" else model.segment(frame)
                output.append(logits.argmax(1).cpu() if finish else logits)
            if stereo:
                z = stereo[0][0](*depth)
                output.append(z.cpu() if finish else z)
        return output

    fixed = inputs()
    conv_macs = model_cost(models, lambda: infer(fixed))
    correlation = 0
    if stereo:
        h, w = config["stereo"]["size"]
        correlation = (h//8)*(w//8)*32*config["stereo"]["shifts"]
    rows, peaks = [], []
    for part in ("model", "end_to_end"):
        for run in range(args.runs):
            for _ in range(args.warmup):
                infer(fixed if part == "model" else inputs(), finish=part == "end_to_end")
            synchronized(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for i in range(args.frames):
                start = perf_counter()
                infer(fixed if part == "model" else inputs(), finish=part == "end_to_end")
                synchronized(device)
                rows.append({"mode": part, "repeat": run+1, "index": i, "ms": (perf_counter()-start)*1000})
            if device.type == "cuda":
                peaks.append({"mode": part, "repeat": run+1, "allocated_bytes": torch.cuda.max_memory_allocated(device)})
    result = {}
    for part in ("model", "end_to_end"):
        times = np.asarray([r["ms"] for r in rows if r["mode"] == part]).reshape(args.runs, args.frames)
        result[part] = {"per_repeat": [{"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
                                        "fps": float(1000/t.mean())} for t in times],
                        "p50_ms": float(np.median(np.percentile(times, 50, axis=1))),
                        "p95_ms": float(np.median(np.percentile(times, 95, axis=1))),
                        "fps": float(np.median(1000/times.mean(axis=1)))}
    macs = conv_macs+correlation
    payload = {"stats": result, "device": args.device,
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "precision": "fp16_amp" if args.fp16 else "fp32", "frame": frame,
               "parameters": sum(p.numel() for model in models for p in model.parameters()),
               "fp32_parameter_bytes": sum(p.numel()*p.element_size() for model in models for p in model.parameters()),
               "checkpoint_bytes": sum(Path(p).stat().st_size for p in args.checkpoints),
               "conv_linear_macs": conv_macs, "stereo_correlation_macs": correlation, "reported_flops": 2*macs,
               "mac_coverage": "Conv2d/ConvTranspose2d/Linear and stereo dot products. Functional wavelet, interpolation, pooling, BN, activations and elementwise ops are outside this count.",
               "peak_cuda": peaks, "warmup": args.warmup, "frames": args.frames, "repeats": args.runs}
    args.out.mkdir(parents=True)
    json_write(args.out/"summary.json", payload)
    with (args.out/"timing.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--image")
    parser.add_argument("--stereo-zip")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    profile(parser.parse_args())
