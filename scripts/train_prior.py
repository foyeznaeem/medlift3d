#!/usr/bin/env python
"""Train the 2-D axial-slice diffusion prior.

Built for a 12-hour Kaggle session: `--max-hours` stops cleanly before the
notebook is killed, and a rerun resumes from `last.pt` automatically.

The prior is unconditional unless `--cond-key` names a conditioning volume
stored in the dataset. That volume must be one that also exists at inference (a
CGLS or FBP initialisation) -- never the clean target, which would train the
network to copy its conditioning channel.

    python scripts/train_prior.py --data data/phantom --out runs/prior --max-hours 11
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from medlift3d.datasets import SliceDataset, default_dirs, load_split, verify_split
from medlift3d.diffusion import DiffusionConfig, GaussianDiffusion
from medlift3d.logging_utils import CsvLogger
from medlift3d.prior2d import UNet2D, UNetConfig
from medlift3d.utils import load_checkpoint, pick_device, save_checkpoint, seed_everything


@torch.no_grad()
def validate(diff, loader, device, n_batches=None, seed=1234):
    """Validation loss with FIXED noise and timesteps.

    Resampling noise every epoch makes the val curve so noisy that "best" is
    chosen by luck. Fixing them per batch index makes epochs comparable.
    """
    diff.eval()
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        if n_batches and i >= n_batches:
            break
        x = batch["x"].to(device)
        cond = batch["cond"].to(device) if "cond" in batch else None
        g = torch.Generator(device="cpu").manual_seed(seed + i)
        t = torch.randint(0, diff.cfg.timesteps, (x.shape[0],), generator=g).to(device)
        noise = torch.randn(x.shape, generator=g).to(device)
        x_t = diff.q_sample(x, t, noise)
        pred = diff.net(x_t, t, cond)
        target = noise if diff.cfg.objective == "eps" else x
        tot += float(torch.nn.functional.mse_loss(pred, target).detach()) * x.shape[0]
        n += x.shape[0]
    diff.train()
    return tot / max(n, 1)


def main():
    d = default_dirs()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=d["data"] / "phantom")
    ap.add_argument("--out", type=Path, default=d["out"] / "prior")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="slices per forward pass; this is what costs VRAM. "
                         "8 at 256x256 needs >15 GB and OOMs a T4")
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="accumulate this many batches before stepping, so the "
                         "effective batch is batch-size * grad-accum without "
                         "the memory of one big batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base-dim", type=int, default=64)
    ap.add_argument("--dim-mults", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--objective", default="eps", choices=["eps", "x0"])
    ap.add_argument("--cfg-drop", type=float, default=0.1)
    ap.add_argument("--cond-key", default=None,
                    help="name of a conditioning volume stored in each case "
                         "(e.g. a CGLS initialisation). Omit for an "
                         "unconditional prior, which is the default: data "
                         "consistency then supplies all patient specificity.")
    ap.add_argument("--amp", action="store_true", default=None)
    ap.add_argument("--multi-gpu", action="store_true",
                    help="split each batch across every visible GPU. Kaggle "
                         "gives two T4s and only one is used otherwise. Raise "
                         "--batch-size alongside it: the batch is divided, so "
                         "8 across two cards costs what 4 did on one")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly after this long (Kaggle sessions die at 12h)")
    ap.add_argument("--save-every", type=int, default=500, help="steps between checkpoints")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    use_amp = (device.type == "cuda") if args.amp is None else args.amp
    args.out.mkdir(parents=True, exist_ok=True)

    split = load_split(args.data / "splits.csv")
    verify_split(split)           # leakage is checked, never assumed
    tr = SliceDataset(args.data, split["train"], cond_key=args.cond_key)
    va = SliceDataset(args.data, split["val"], cond_key=args.cond_key)
    cache_gb = (tr.cache_bytes + va.cache_bytes) / 2 ** 30
    print(f"batch {args.batch_size} x {args.grad_accum} accum "
          f"= effective {args.batch_size * args.grad_accum}")
    print(f"device={device} amp={use_amp}  train slices={len(tr)}  val slices={len(va)}"
          + (f"  slice cache {cache_gb:.2f} GB" if cache_gb else "  (uncached)"))
    print(f"train cases={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])} (test never seen by the prior)")

    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=(device.type == "cuda"),
                       drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, persistent_workers=args.workers > 0)

    ucfg = UNetConfig(base_dim=args.base_dim, dim_mults=tuple(args.dim_mults),
                      cond_ch=1 if args.cond_key else 0)
    dcfg = DiffusionConfig(timesteps=args.timesteps, objective=args.objective,
                           cfg_drop_prob=args.cfg_drop)
    diff = GaussianDiffusion(UNet2D(ucfg), dcfg).to(device)
    if args.multi_gpu:
        diff.parallelize()
        if diff.n_devices > 1:
            names = [torch.cuda.get_device_name(i) for i in range(diff.n_devices)]
            print(f"multi-GPU: batch split across {diff.n_devices} x {names[0]}")
        else:
            print("--multi-gpu asked for, but only one GPU is visible; "
                  "running on it alone")
    print(f"UNet {diff.model.n_params/1e6:.1f}M params  cond_ch={ucfg.cond_ch}  "
          f"objective={dcfg.objective}")

    opt = torch.optim.AdamW(diff.model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    step, start_epoch, best = 0, 0, float("inf")
    last_path = args.out / "last.pt"
    if last_path.exists() and not args.no_resume:
        ck = load_checkpoint(last_path, map_location=device)
        diff.load_state(ck)                      # refuses a config mismatch
        opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        step, start_epoch, best = ck["step"], ck["epoch"], ck.get("best", float("inf"))
        print(f"resumed from {last_path} at epoch {start_epoch} step {step} "
              f"(best val {best:.5f})")

    log = CsvLogger(args.out / "log.csv")
    t_start = time.time()
    stop = False

    def checkpoint(epoch, tag="last"):
        save_checkpoint(args.out / f"{tag}.pt", optimizer=opt.state_dict(),
                        scaler=scaler.state_dict() if use_amp else None,
                        step=step, epoch=epoch, best=best, args=vars(args),
                        **diff.state())

    for epoch in range(start_epoch, args.epochs):
        diff.train()
        bar = tqdm(dl_tr, desc=f"epoch {epoch+1}/{args.epochs}")
        run, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for micro, batch in enumerate(bar, start=1):
            x = batch["x"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True) if "cond" in batch else None
            with torch.amp.autocast(device.type, enabled=use_amp):
                # Divided so the accumulated gradient equals what one pass over
                # batch_size * grad_accum samples would have produced.
                loss = diff.loss(x, cond) / args.grad_accum
            scaler.scale(loss).backward()

            # `seen` counts micro-batches and `run` sums undivided losses, so
            # run/seen stays the mean loss per slice whatever grad_accum is.
            seen += 1
            run += float(loss.detach()) * args.grad_accum

            if micro % args.grad_accum:
                continue                      # keep accumulating, do not step

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(diff.model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            step += 1
            bar.set_postfix(loss=f"{run/seen:.4f}", step=step)
            if step % args.save_every == 0:
                checkpoint(epoch)
            if args.max_hours and (time.time() - t_start) / 3600 > args.max_hours:
                print(f"\n--max-hours reached at step {step}; checkpointing and exiting")
                stop = True
                break

        vloss = validate(diff, dl_va, device, n_batches=40)
        tloss = run / max(1, seen)
        log.log(epoch=epoch + 1, step=step, train_loss=tloss, val_loss=vloss)
        print(f"epoch {epoch+1}: train {tloss:.5f}  val {vloss:.5f}")

        if vloss < best:
            best = vloss
            checkpoint(epoch + 1, tag="best")
            print(f"  new best val {best:.5f} -> best.pt")
        checkpoint(epoch + 1)
        if stop:
            break

    log.plot("epoch", ["train_loss", "val_loss"])
    print(f"\ndone. best val {best:.5f}. checkpoints in {args.out}")
    if stop:
        print("Rerun the same command to resume from last.pt.")


if __name__ == "__main__":
    main()
