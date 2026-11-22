"""Command line for the semantic experiment matrix and its supervised stereo extensions."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import platform
import sys

import cv2
import torch

from multitask.data import SEQUENCES, SemanticData, StereoData, StereoSpec
from multitask.engine import digest, evaluate, json_write, read, restore_model, run_training
from multitask.models import Group, WIDTHS, all_groups


def code_digest():
    import hashlib
    root = Path(__file__).resolve().parent
    files = [Path(__file__).resolve(), *sorted((root/"multitask").glob("*.py")),
             *sorted((root/"seg_backbone/models").glob("*.py"))]
    return hashlib.sha256("".join(digest(p) for p in files).encode()).hexdigest()


def planned_jobs():
    pipe, bio = all_groups("pipe"), all_groups("bio")
    for g in (*pipe, *bio):
        for fold, seed in itertools.product(range(5) if g.task == "pipe" else [0], (42, 43, 44)):
            yield {"stage": "seg", "task": g.task, "group": g.name, "width": g.width, "fold": fold, "seed": seed}
    yield {"stage": "depth", "task": None, "group": None, "width": None, "fold": 0, "seed": 42}
    for g in (*pipe, *bio):
        yield {"stage": "joint", "task": g.task, "group": g.name, "width": g.width, "fold": 0, "seed": 42}


def job_path(root, job):
    stage = job["stage"]
    if stage == "depth":
        return Path(root)/"depth"/f"baseline-s{job['seed']}"
    return (Path(root)/stage/job["task"]/f"{job['group']}-W{job['width']}"
            /f"f{job['fold']}-s{job['seed']}")


def plan(args):
    jobs = list(planned_jobs())
    counts = {stage: sum(j["stage"] == stage for j in jobs) for stage in ("seg", "depth", "joint")}
    assert counts == {"seg": 1536, "depth": 1, "joint": 160}, counts
    json_write(args.out, {"counts": counts, "total": len(jobs), "jobs": jobs})
    print(counts, "total", len(jobs))


def pick_source(args, group):
    path = args.pipe_source if group.task == "pipe" else args.bio_source
    if not path:
        raise ValueError("Provide the source for the relevant segmentation task")
    return path


def sequences(args):
    parts = {"train": args.stereo_train, "val": args.stereo_val, "test": args.stereo_test}
    flat = [seq for part in parts.values() for seq in part]
    if set(flat) != set(SEQUENCES) or len(flat) != len(set(flat)):
        raise ValueError("FLSea train/val/test sequence lists must partition the four sequences")
    return parts


def choose_weight(args, key):
    if getattr(args, "weight", None):
        return Path(args.weight).resolve(strict=True)
    if args.weights:
        base = Path(args.weights).resolve()
        path = read(base).get(key)
        if path:
            return (base.parent/path).resolve(strict=True)
    if not args.scratch:
        raise ValueError(f"Missing pinned initialization for {key}. Supply weights or explicitly use --scratch.")
    return None


def train(args):
    group = Group(args.task, args.group, args.width) if args.stage != "depth" else None
    if group is not None and group.width not in WIDTHS:
        raise ValueError(f"Training width must be one of {WIDTHS}")
    fold = args.fold if group and group.task == "pipe" else 0
    if group is not None and not 0 <= fold < 5:
        raise ValueError("Pipeline fold must be in [0,4]")
    sem = SemanticData(group.task, pick_source(args, group), "train", fold, args.limit_sem,
                       True, args.seed) if group else None
    stereo = spec = parts = None
    if args.stage != "seg":
        if not args.stereo_zip or not args.direction or not args.depth_kind:
            raise ValueError("Stereo training requires an archive and explicit calibration direction and label convention")
        parts = sequences(args)
        spec = StereoSpec(tuple(args.stereo_size), args.direction, args.depth_kind,
                          args.min_depth, args.max_depth, args.shifts)
        stereo = StereoData(args.stereo_zip, parts["train"], spec, args.limit_stereo)
    parent = None
    if args.stage == "joint":
        if not args.parent:
            raise ValueError("Joint training requires the completed segmentation checkpoint")
        parent = Path(args.parent).resolve(strict=True)
    weights = None if parent else choose_weight(args, group.weight_key if group else "efficientnet_b0")
    batch = args.batch
    effective = args.effective_batch or (8 if group is None or group.task == "pipe" else 16)
    if (args.epochs < 1 or batch < 2 or effective < batch or effective % batch
            or args.steps < 0 or args.limit_sem < 0 or args.limit_stereo < 0
            or args.depth_weight <= 0):
        raise ValueError("Use batch >= 2, divisible effective batch and positive training budget")
    bio = group is not None and group.task == "bio"
    meta = {"stage": args.stage, "group": asdict(group) if group else None, "seed": args.seed, "fold": fold,
            "stereo": asdict(spec) if spec else None, "stereo_splits": parts,
            "semantic_signature": sem.signature if sem is not None else None,
            "stereo_signature": stereo.signature if stereo is not None else None,
            "epochs": args.epochs, "steps": args.steps, "batch": batch, "effective_batch": effective,
            "lr": 3e-4 if bio else 1e-4, "min_lr": 1e-5 if bio else 1e-6,
            "depth_weight": args.depth_weight, "device": args.device,
            "initial_sha256": digest(weights) if weights else None,
            "parent_sha256": digest(parent) if parent else None,
            "code_sha256": code_digest(),
            "torch": str(torch.__version__), "platform": platform.platform()}
    run_training(meta, args.out, sem, stereo, weights, parent, args.resume, args.stop_after)


def evaluate_command(args):
    device = torch.device(args.device)
    model, meta = restore_model(args.checkpoint, device)
    # The eight-width release changed only the experiment matrix and CLI selection.
    previous_matrix_revision = "02fe5ac8aa871ed64eeae98f7a1edd5982739b93cd245941edf116f117b625d4"
    if meta["code_sha256"] not in (code_digest(), previous_matrix_revision):
        raise ValueError("Checkpoint and evaluation program belong to different code revisions")
    group = Group(**meta["group"]) if meta["group"] else None
    available_semantic = group and (args.pipe_source if group.task == "pipe" else args.bio_source)
    sem = SemanticData(group.task, pick_source(args, group), args.sem_split, meta["fold"], args.limit_sem
                       ) if available_semantic else None
    stereo = StereoData(args.stereo_zip, meta["stereo_splits"][args.stereo_part],
                        StereoSpec(**meta["stereo"]), args.limit_stereo
                        ) if meta["stereo"] and args.stereo_zip else None
    if sem is None and stereo is None:
        raise ValueError("Supply at least one source for a task supported by the checkpoint")
    stats, rows = evaluate(model, meta, sem, stereo, device)
    args.out.mkdir(parents=True)
    json_write(args.out/"summary.json", {**stats, "checkpoint_sha256": digest(args.checkpoint),
               "group": meta["group"], "fold": meta["fold"], "seed": meta["seed"], "stage": meta["stage"],
               "device": args.device, "semantic_split": args.sem_split if sem is not None else None,
               "stereo_split": args.stereo_part if stereo is not None else None})
    with (args.out/"predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    print(stats, flush=True)


def run_plan(args):
    from types import SimpleNamespace
    rows = read(args.plan)["jobs"]
    jobs = rows[args.start:None if not args.count else args.start+args.count]
    if not jobs:
        raise ValueError("Empty plan slice")
    status = []
    for job in jobs:
        out = job_path(args.out, job)
        finished = (out/"model.pt").is_file()
        stage = job["stage"]
        parent = job_path(args.out, {**job, "stage": "seg"})/"model.pt" if stage == "joint" else None
        current = SimpleNamespace(**{**vars(args), **job, "out": out, "parent": parent,
            "resume": out.exists(), "weight": None, "epochs": args.epochs if stage == "seg" else args.stereo_epochs,
            "stop_after": None})
        try:
            # Validate the locked configuration on resume even for completed jobs.
            train(current)
            record = {"job": job, "status": "already_complete" if finished else "complete", "path": str(out)}
        except Exception as exc:
            record = {"job": job, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        status.append(record)
        json_write(Path(args.out)/f"batch-{args.start}.json", status)
    if any(row["status"] == "failed" for row in status):
        raise SystemExit(1)


def source_options(parser):
    parser.add_argument("--pipe-source", help="Pipeline mask CSV manifest")
    parser.add_argument("--bio-source", help="SUIM mask CSV manifest or NPY cache")
    parser.add_argument("--stereo-zip", help="Local FLSea Stereo ZIP archive")
    parser.add_argument("--limit-sem", type=int, default=0, help="Evenly sample at most this many semantic frames")
    parser.add_argument("--limit-stereo", type=int, default=0, help="Maximum frames per FLSea sequence")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)


def training_options(parser):
    source_options(parser)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", help="JSON mapping each backbone family to local pretrained weights")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--effective-batch", type=int)
    parser.add_argument("--steps", type=int, default=0, help="Limit steps per epoch for a pilot")
    parser.add_argument("--direction", choices=("stored-ltr", "stored-rtl"))
    parser.add_argument("--depth-kind", choices=("z", "range"))
    parser.add_argument("--stereo-size", nargs=2, type=int, default=(288, 512))
    parser.add_argument("--shifts", type=int, default=64, help="Number of disparity hypotheses at stride eight")
    parser.add_argument("--min-depth", type=float, default=.5)
    parser.add_argument("--max-depth", type=float, default=15.)
    parser.add_argument("--depth-weight", type=float, default=1.)
    parser.add_argument("--stereo-train", nargs="+", default=["canyon1", "rock_garden1"])
    parser.add_argument("--stereo-val", nargs="+", default=["rock_garden2"])
    parser.add_argument("--stereo-test", nargs="+", default=["canyon2"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=plan)
    p = sub.add_parser("train")
    training_options(p)
    p.add_argument("--stage", choices=("seg", "depth", "joint"), required=True)
    p.add_argument("--task", choices=("pipe", "bio"))
    p.add_argument("--group")
    p.add_argument("--width", type=int, choices=WIDTHS)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight", help="Local pretrained checkpoint override")
    p.add_argument("--parent", help="Segmentation model.pt for joint initialization")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int)
    p.set_defaults(func=train)
    p = sub.add_parser("eval")
    source_options(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sem-split", choices=("val", "test"), default="test")
    p.add_argument("--stereo-part", choices=("val", "test"), default="test")
    p.set_defaults(func=evaluate_command)
    p = sub.add_parser("batch")
    training_options(p)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--stereo-epochs", type=int, required=True)
    p.set_defaults(func=run_plan)
    args = parser.parse_args()
    if hasattr(args, "device"):
        if args.threads < 1 or args.limit_sem < 0 or args.limit_stereo < 0:
            parser.error("Invalid thread or sample count")
        cv2.setNumThreads(args.threads)
        torch.set_num_threads(args.threads)
    if args.command == "batch" and (args.start < 0 or args.count <= 0 or args.stereo_epochs <= 0):
        parser.error("Batch execution requires a non-negative start and positive count and epochs")
    args.func(args)


if __name__ == "__main__":
    main()
