# Running MedLift-3D on a rented GPU

Short answer: **yes, this works.** Claude Code has shell access on your machine,
so it can drive a rented box over SSH — sync the repo, run the gates on real
hardware, launch training detached, poll progress, run every experiment, and
pull results back. Nothing about the design assumes a local GPU.

This works for vast.ai, RunPod, Lambda, Paperspace or any host that gives you
SSH. Instructions below use vast.ai because that is what you asked about.

---

## Division of labour

**You do these three things** (they need your account and your money, so they
are not mine to do):

1. Create a vast.ai account and add credit.
2. Paste the public key from `./scripts/remote.sh keygen` into
   **Account → SSH Keys**.
3. Rent an instance and copy its SSH connection string from the instance card.

**Then I do the rest:** environment setup, code sync, gate verification on the
real GPU, benchmark, training, reconstruction, all experiments, fetching results
and writing them up.

## What to rent

The workload is a 48M-parameter 2D UNet plus a memory-bandwidth-bound ray
caster. It does **not** need an A100 or H100 — paying for one is wasted money.

| | Recommendation |
|---|---|
| GPU | RTX 3090 / 4090 / A5000 — 24 GB. 16 GB works; 24 GB lets you raise the batch size. |
| Disk | 60 GB is plenty (a 60-case phantom set at 256³ is ~3 GB) |
| Type | **On-demand, not interruptible.** Training resumes from checkpoints if killed, but spot preemption mid-experiment wastes more than it saves. |
| Image | Any PyTorch/CUDA image. `pytorch/pytorch:latest` or vast's PyTorch template. |

Check current rates on the site; as a rough guide a 3090/4090 is usually in the
region of a few tens of cents per hour, and the full programme below is on the
order of single-digit dollars. `gpu_benchmark.py --rate` turns that into a real
number before you commit.

## Sequence

```bash
# --- once ---
./scripts/remote.sh keygen              # then paste the key into vast.ai

# --- each new instance ---
export MEDLIFT_HOST='-p 12345 root@ssh5.vast.ai'   # from the instance card
export MEDLIFT_RATE=0.40                            # $/hr, for cost reporting

./scripts/remote.sh setup               # GPU, disk, torch, installs tmux
./scripts/remote.sh sync                # rsync + pip install -e .

# --- verify before spending ---
./scripts/remote.sh run 'python scripts/run_gates.py'
./scripts/remote.sh run 'python scripts/gpu_benchmark.py --rate 0.40'
```

`gpu_benchmark.py` measures fp/bp time, training step time and peak VRAM on the
actual card, then prices the whole programme. **Read its output before launching
anything long** — that is the difference between a $5 run and discovering at
hour six that it needed 40.

```bash
# --- the programme ---
./scripts/remote.sh launch data  'python scripts/make_phantoms.py --n-cases 60 \
    --shape 256 256 256 --views-a 16 32 --views-b 15 --device cuda \
    --out /workspace/data/phantom'

./scripts/remote.sh launch train 'python scripts/train_prior.py \
    --data /workspace/data/phantom --out /workspace/medlift3d/runs/prior \
    --epochs 60 --batch-size 16 --amp --workers 4'

./scripts/remote.sh watch train         # poll; the job is detached
./scripts/remote.sh status              # GPU util, running jobs, disk
```

Then reconstruction, evaluation and the three experiments, each launched the
same way. Finally:

```bash
./scripts/remote.sh fetch               # metrics, figures, NIfTI, logs
./scripts/remote.sh fetch-all           # + checkpoints (large)
./scripts/remote.sh cost
```

**Destroy the instance in the web UI when done.** vast.ai bills while the
instance *exists*, not only while it computes. A forgotten idle box is the most
common way to waste money here.

## Why everything runs detached

Every long job goes through `remote.sh launch`, which starts it in a tmux
session with output tee'd to `logs/<name>.log`. Consequences that matter:

- A dropped SSH connection, a closed laptop, or a killed terminal costs nothing.
- My own tool calls time out after a few minutes, so I *cannot* sit inside a
  six-hour training run. I launch it, then poll with `watch`. This is the only
  arrangement that works, and it is also the robust one.
- `train_prior.py --max-hours N` still applies. It is there for Kaggle's 12-hour
  cap, but it doubles as insurance: if the instance dies, rerunning the same
  command resumes from `last.pt` with optimiser, scaler, step and best-val
  intact.

## How I work across a long run

I act when you message me — I do not run continuously in the background. So a
multi-hour run looks like:

1. You say go. I sync, verify, benchmark, and launch training detached.
2. I report the benchmark numbers and the expected finish time.
3. You ping me later ("check on it"). I poll, report, and launch the next stage.

If you would rather I self-pace, `/loop` makes me check in on an interval
(`/loop 30m check the training run and launch the next stage when it finishes`).
That costs tokens per check, so a long interval is better than a short one.

## Cost guardrails

Because this spends your money, I will:

- Run `gpu_benchmark.py` and show you the projected cost **before** launching
  the long stages.
- Tell you the burn rate and elapsed spend whenever I check in.
- Not create or destroy instances without asking, even if you give me a
  `vastai` API key.
- Flag it if a stage is running well over its estimate rather than letting it
  quietly run.

If you want tighter control, set a spending limit in the vast.ai UI — that is a
hard stop I cannot exceed by mistake.

## Optional: the vastai CLI

`pip install vastai` gives programmatic search/create/destroy. With an API key
configured I could pick an instance by price and specs, launch it, and tear it
down when the run finishes. I would still confirm each create and destroy with
you first — those are the actions that cost money and lose data.

## Real data

The full LIDC-IDRI download is ~124 GB and the TCIA client is slow, so pulling
it onto a metered instance is usually a poor trade. Better options, in order:

1. **Phantoms** (`make_phantoms.py`) — no download, everything runs today, and
   they exercise every code path. Good enough to validate the method and produce
   the MDVC and hallucination curves.
2. **A Kaggle-hosted LUNA/LIDC subset**, downloaded once to a vast.ai volume:
   `prepare_lidc.py --source dir`. Note this route gives volumes **without**
   nodule contours — usable for training the prior, not for Dice or volume
   error.
3. **Full LIDC-IDRI via `pylidc`** on a box with a persistent volume, for the
   final clinical numbers. Only LIDC's four-reader contours support the nodule
   metrics.

## Failure modes seen in practice

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied (publickey)` | key not registered, or instance created before you added it | re-add the key, then recreate the instance |
| `setup` shows `NO GPU VISIBLE` | CPU-only instance, or driver mismatch | destroy and rent a different one; do not debug a bad box |
| `launch` says session already running | previous job still alive | `remote.sh stop <name>` or use another name |
| training OOM | batch too large for the card | lower `--batch-size`; `gpu_benchmark.py` reports peak VRAM |
| everything is slow | you rented a card with poor memory bandwidth | the projector is bandwidth-bound; check `status` for GPU utilisation |
| disk full mid-run | checkpoints plus data | `remote.sh status` shows disk; fetch and prune |
