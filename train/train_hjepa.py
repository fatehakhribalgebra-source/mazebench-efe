"""Train the hierarchical JEPA predictors on encoded MazeBench frames.

Each level is an independent predictor over the same latent space, so they
train in separate passes. The encoder is never loaded here -- it ran once in
``train/frames.py``.

Every validation reports the **identity baseline** beside the model: the loss
you get by predicting that nothing changes. In a world where one move alters a
small part of a frame, identity is a genuinely strong predictor, and a model
that does not clearly beat it has learned nothing about dynamics no matter how
small its absolute loss looks. This is the same trap as scoring ``tile_acc`` on
a grid that mostly does not move, which this project fell into once already.

    python -m train.train_hjepa --latents runs/vision/latents --levels 3 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.hjepa import HJEPA, HJEPAConfig
from train.frames import LatentEpisodes, WindowSpec, split_episodes


def identity_baseline(batch: dict) -> torch.Tensor:
    """Loss of predicting the last context latent for every target."""
    last = batch["context"][:, -1:].expand_as(batch["targets"])
    return torch.nn.functional.mse_loss(last, batch["targets"])


def run_epoch(model, level, loader, optimiser, device, train: bool, max_batches=None,
              amp: bool = False):
    """One pass over ``loader``.

    ``amp`` runs the predictor under bfloat16 autocast. bf16 rather than fp16
    because it keeps fp32's exponent range, so no gradient scaler is needed and
    the loss cannot silently underflow -- which matters here, where the losses
    are already small numbers being compared against a baseline. The identity
    baseline is computed outside autocast so the number the gate is judged on is
    always full precision and comparable across runs.
    """
    model.train(train)
    totals = {"loss": 0.0, "first": 0.0, "last": 0.0, "identity": 0.0, "n": 0}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=amp and str(device).startswith("cuda")):
                out = model.loss(batch["context"], batch["actions"],
                                 batch["targets"], level=level)
            out = {k: (v.float() if torch.is_tensor(v) else v)
                   for k, v in out.items()}
        if train:
            optimiser.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.levels[level].parameters(), 1.0)
            optimiser.step()
        totals["loss"] += out["loss"].item()
        totals["first"] += out["first"].item()
        totals["last"] += out["last"].item()
        totals["identity"] += identity_baseline(batch).item()
        totals["n"] += 1
    n = max(totals.pop("n"), 1)
    return {k: v / n for k, v in totals.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--latents", required=True, help="directory of encoded episode .npz files")
    p.add_argument("--out", default="runs/hjepa")
    p.add_argument("--levels", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--context", type=int, default=2)
    p.add_argument("--rollout", type=int, default=4)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--max-batches", type=int, default=None, help="cap batches per epoch")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--amp", action="store_true",
                   help="bfloat16 autocast for the predictor; identity baseline "
                        "stays fp32 so the gate number is unaffected")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = HJEPAConfig(
        n_levels=args.levels, context=args.context, rollout_steps=args.rollout,
        depth=args.depth, heads=args.heads,
    )
    model = HJEPA(config).to(args.device)
    train_eps, val_eps = split_episodes(args.latents, args.val_frac, args.seed)
    print(f"episodes: {len(train_eps)} train, {len(val_eps)} val")

    report = {"args": vars(args), "levels": {}}
    for level, stride in enumerate(model.strides):
        spec = WindowSpec(stride=stride, context=args.context, rollout=args.rollout)
        try:
            train_set = LatentEpisodes(args.latents, spec, train_eps)
            val_set = LatentEpisodes(args.latents, spec, val_eps)
        except ValueError as error:
            print(f"\nlevel {level + 1} (stride {stride}): SKIPPED -- {error}")
            report["levels"][str(level + 1)] = {"skipped": str(error)}
            continue

        print(
            f"\nlevel {level + 1} (stride {stride}): {len(train_set)} train windows, "
            f"{len(val_set)} val, needs {spec.span()} frames per episode"
            + (f", {train_set.skipped_episodes} episodes too short" if train_set.skipped_episodes else "")
        )
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
            drop_last=True,
        )
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
        optimiser = torch.optim.AdamW(
            model.levels[level].parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        history, best = [], float("inf")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, level, train_loader, optimiser, args.device, True,
                           args.max_batches, args.amp)
            va = run_epoch(model, level, val_loader, optimiser, args.device, False,
                           args.max_batches, args.amp)
            ratio = va["loss"] / max(va["identity"], 1e-9)
            history.append({"epoch": epoch, "train": tr, "val": va, "vs_identity": ratio})
            print(
                f"  ep {epoch:3d}  train {tr['loss']:.4f}  val {va['loss']:.4f}  "
                f"identity {va['identity']:.4f}  ratio {ratio:.3f}  "
                f"step1 {va['first']:.4f}  step{args.rollout} {va['last']:.4f}  "
                f"({time.time() - t0:.0f}s)"
            )
            if va["loss"] < best:
                best = va["loss"]
                # Per level: a single best.pt was overwritten by each level in
                # turn, so it ended up holding only the last level's best and
                # silently misrepresented the others.
                torch.save(
                    {"config": config.__dict__, "state": model.state_dict(), "level": level},
                    out / f"best_level{level + 1}.pt",
                )
        report["levels"][str(level + 1)] = {
            "stride": stride, "history": history, "best_val": best,
            "beats_identity": history[-1]["vs_identity"] < 1.0,
        }

    torch.save({"config": config.__dict__, "state": model.state_dict()}, out / "final.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print("\n--- gate ---")
    for name, info in report["levels"].items():
        if "skipped" in info:
            print(f"  level {name}: no data")
        else:
            verdict = "PASS" if info["beats_identity"] else "FAIL (no better than predicting no change)"
            print(f"  level {name}: ratio-to-identity {info['history'][-1]['vs_identity']:.3f}  {verdict}")
    print(f"\nwrote {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
