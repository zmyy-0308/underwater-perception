"""One resumable, fixed-budget distillation trial using original networks."""
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from experiment import ROOT, ROLES, Samples, digest, evaluate, load, restore_state, save, torchify_state, write_json
from objectives import Objective, RECIPES


def train(args):
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    device = torch.device(args.device)
    recipe = RECIPES[args.recipe]
    role = ROLES[recipe.student]
    out = args.out_root / args.recipe / f"f{args.fold}-s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    student, student_meta = load(role, args.fold, device)
    teacher = None
    if recipe.logit or recipe.feature:
        teacher, teacher_meta = load(ROLES[recipe.teacher], args.fold, device)
        teacher.eval().requires_grad_(False)
        assert teacher_meta["fold"] == student_meta["fold"]
    # Projection initialization is paired independently from model construction.
    torch.manual_seed(seed + 30000)
    objective = Objective(recipe, {4: 32, 8: 64}).to(device)
    optimizer = torch.optim.Adam([*student.parameters(), *objective.parameters()],
                                 lr=1e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_set = Samples(args.fold, "train", seed, args.limit)
    test_set = Samples(args.fold, "test", seed, args.limit)
    meta = {
        "recipe_name": args.recipe, "recipe": asdict(recipe), "fold": args.fold,
        "seed": seed, "epochs": args.epochs, "batch": 2, "effective_batch": 8,
        "initial_sha256": digest(ROOT / f"assets/weights/{role}-f{args.fold}.pt"),
        "teacher_sha256": digest(ROOT / f"assets/weights/{ROLES[recipe.teacher]}-f{args.fold}.pt") if teacher else None,
        "original_student_meta": student_meta, "limit": args.limit,
        "code_sha256": digest(Path(__file__)) + digest(ROOT / "objectives.py"),
    }
    meta = json.loads(json.dumps(meta))
    if (out / "config.json").exists():
        assert json.loads((out / "config.json").read_bytes()) == meta, "Resume config differs"
    write_json(out / "config.json", meta)
    if (out / "summary.json").exists():
        return
    history = []
    if (out / "resume.pt").exists():
        state = torch.load(out / "resume.pt", map_location="cpu", weights_only=True)
        assert state["meta"] == meta
        student.model.load_state_dict(state["model"], strict=True)
        objective.load_state_dict(state["objective"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_state(state["rng"])
        history = state["history"]
    for epoch in range(len(history), args.epochs):
        torch.manual_seed(seed + 60000 + epoch)
        start = perf_counter()
        student.train()
        train_set.epoch(epoch)
        generator = torch.Generator().manual_seed(seed + 20000 + epoch)
        loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True,
                                             drop_last=True, generator=generator, num_workers=0)
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(4)
        if amp:
            torch.cuda.reset_peak_memory_stats(device)
        for step, (x, y, _) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                if teacher:
                    with torch.no_grad():
                        target_logits, target_features = teacher(x)
                else:
                    target_logits, target_features = None, None
                logits, features = student(x)
                loss, components = objective(logits, features, y, target_logits, target_features,
                                             epoch / max(1, args.epochs - 1))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite distillation loss")
            block = min(4, len(loader) - (step // 4) * 4)
            scaler.scale(loss / block).backward()
            totals += np.array([float(loss.detach()), *(float(c.detach()) for c in components)])
            if (step + 1) % 4 == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if amp:
            torch.cuda.synchronize()
        history.append({
            "epoch": epoch + 1, "seconds": perf_counter() - start,
            **dict(zip(("loss", "supervised", "pixel_kd", "feature_kd"), (totals / len(loader)).tolist())),
            "lr": optimizer.param_groups[0]["lr"],
            "peak_training_bytes": torch.cuda.max_memory_allocated() if amp else 0,
        })
        schedule.step()
        save(out / "resume.pt", {"meta": meta, "model": student.model.state_dict(),
             "objective": objective.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": schedule.state_dict(), "scaler": scaler.state_dict(),
             "history": history, "rng": torchify_state()})
        write_json(out / "history.json", history)
        print(args.recipe, args.fold, seed, history[-1], flush=True)
        if args.stop_after and epoch + 1 >= args.stop_after:
            return
    result, images = evaluate(student, test_set, device)
    restored, _ = load(role, args.fold, device)
    restored.model.load_state_dict(student.model.state_dict(), strict=True)
    save(out / "model.pt", {"meta": student_meta, "epoch": 60,
                           "model": student.model.state_dict(), "distillation": meta})
    result.update(recipe=args.recipe, fold=args.fold, seed=seed,
                  checkpoint_sha256=digest(out / "model.pt"),
                  parameters=sum(p.numel() for p in student.model.parameters()),
                  training_seconds=sum(row["seconds"] for row in history))
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in images:
            stream.write(json.dumps(row) + "\n")
    write_json(out / "summary.json", result)
    (out / "COMPLETE").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs")
    train(parser.parse_args())
