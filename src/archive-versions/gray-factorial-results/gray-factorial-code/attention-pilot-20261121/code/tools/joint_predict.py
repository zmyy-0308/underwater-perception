"""Export co-registered left-view semantic class maps and metric scene depth."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multitask.data import SEQUENCES, StereoData, StereoSpec, rgb_tensor
from multitask.canny import CANNY_GROUPS
from multitask.engine import digest, json_write, restore_model
from joint_profile import semantic_geometry


@torch.inference_mode()
def predict(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    models = [(model.eval(), meta) for model, meta in (restore_model(p, device) for p in args.checkpoints)]
    stereo = [(model, meta) for model, meta in models if meta["stereo"]]
    tasks = [(model, meta) for model, meta in models if meta["group"]]
    tags = [meta["group"]["task"] for _, meta in tasks]
    if len(stereo) != 1 or len(set(tags)) != len(tags):
        raise ValueError("Provide one depth-producing checkpoint and at most one checkpoint per semantic task")
    model, meta = stereo[0]
    spec = StereoSpec(**meta["stereo"])
    data = StereoData(args.stereo_zip, [args.sequence], spec)
    if not 0 <= args.index < len(data):
        raise ValueError("Frame index is outside the sequence")
    left, right, _, id_ = data.raw(args.index)
    left, right = [cv2.remap(im, *maps, cv2.INTER_LINEAR)
                   for im, maps in zip((left, right), data.rectifier.maps)]
    input_left, input_right = [rgb_tensor(im)[None].to(device) for im in (left, right)]
    fb = torch.tensor([data.rectifier.fb], device=device)
    z = model(input_left, input_right, fb)[0].cpu().numpy().astype(np.float32)
    args.out.mkdir(parents=True)
    Image.fromarray(left).save(args.out/"left_rectified.png")
    Image.fromarray(z).save(args.out/"depth_z_m.tif")
    normalized = ((z-spec.min_depth)/(spec.max_depth-spec.min_depth)).clip(0, 1)
    display = cv2.applyColorMap((normalized*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)).save(args.out/"depth_preview.png")
    for semantic, record in tasks:
        task = record["group"]["task"]
        sample, (x, y, w, h) = semantic_geometry(left, task, record["group"]["name"] in CANNY_GROUPS)
        sample = sample.to(device)
        out = semantic(sample) if record["stage"] == "seg" else semantic.segment(sample)
        mask = out.argmax(1)[0].cpu().numpy().astype(np.uint8)
        mh, mw = mask.shape
        full = np.full((h, w), 255, dtype=np.uint8)
        full[y:y+mh, x:x+mw] = mask
        full = cv2.resize(full, (spec.size[1], spec.size[0]), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(full).save(args.out/f"{task}_classes.png")
    json_write(args.out/"sample.json", {"id": id_, "size": list(z.shape), "depth_units": "m",
               "depth_axis": "rectified_left_optical_axis", "semantic_outside_crop": 255,
               "tasks": tags, "source_checkpoint_sha256": [digest(p) for p in args.checkpoints]})
    print(id_, z.shape, tags)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    p.add_argument("--stereo-zip", required=True)
    p.add_argument("--sequence", choices=SEQUENCES, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True, type=Path)
    predict(p.parse_args())
