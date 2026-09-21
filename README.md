# MedLift-3D

Volumetric lung nodule assessment from sparse-view and limited-angle chest
projections.

Reconstruct 3D lung anatomy from few 2D projections using a learned anatomical
prior that is **hard-constrained to agree with the measurements**, and report a
**per-voxel uncertainty** so that prior-invented structure is distinguishable
from measured structure. The headline deliverable is not "CT-quality volumes" —
it is nodule volumetry with stated error bounds, plus an explicit map of where
the method stops working.

Clean-room implementation. Design rationale in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md);
the failure modes it exists to avoid are catalogued in [docs/CODE_AUDIT.md](docs/CODE_AUDIT.md).

---

## Quick start

```bash
pip install -e .                       # or: pip install -r requirements.txt

python scripts/run_gates.py            # 1. correctness gates  (~15 s, CPU)
python scripts/make_phantoms.py --n-cases 40 --out data/phantom
python scripts/train_prior.py  --data data/phantom --out runs/prior --max-hours 11
python scripts/reconstruct.py  --data data/phantom --out runs/recon --track A16 \
                               --method diffusion --prior runs/prior/best.pt
python scripts/evaluate.py     --recon runs/recon --data data/phantom
```

No dataset download is required: `make_phantoms.py` generates synthetic chest
phantoms with nodules, so the entire pipeline is runnable and verifiable before
any data agreement is signed. Real data goes through `scripts/prepare_lidc.py`;
everything downstream is identical.

## Run the gates first

Four failure modes in this problem are **silent** — they produce plausible loss
curves and wrong reconstructions. Each is a test, not a judgement call:

| Gate | Asserts | Guards against |
|---|---|---|
| **G1** `test_geometry` | one canonical grid, affine never dropped | comparing volumes on different grids |
| **G2** `test_adjoint` | `fp` is differentiable; `bp` is its exact adjoint | a projection loss with **no gradient** |
| **G3** `test_units` | water cylinder line integral = `mu_water · 2R` | HU/density confusion, scale errors |
| **G4** `test_metrics` | a *perfect* reconstruction scores perfectly | metrics that cannot detect anything |

Plus: no data leakage, diffusion train/sample config agreement, ROI decomposition
exactness, Gaussian field correctness, and that data consistency measurably
drives the projection residual down. `python scripts/run_gates.py` runs all nine
suites. **Do not start training until every gate passes.**

## How it works

```
 measured projections p  (V × H × W)
            │
   ┌────────▼─────────┐
   │ 1  Initialiser   │  CGLS / SIRT-TV                     → coarse mu, 256³ @ 1.5 mm
   └────────┬─────────┘
   ┌────────▼─────────┐
   │ 2  Prior         │  2D axial-slice diffusion (~48M params, ~7 GB to train)
   └────────┬─────────┘
   ┌────────▼─────────┐
   │ 3  DC solver     │  prior denoise ⇄ A^T(A x − p) + λ_z·TV_z,  ×K samples
   └────────┬─────────┘        mean = reconstruction · std = uncertainty
   ┌────────▼─────────┐
   │ 4  ROI refine    │  128³ @ 1.0 mm, Gaussian or voxel  [ablated, objective O5]
   └────────┬─────────┘
            ▼
   NIfTI + uncertainty + nodule mask + DICOM SEG/SR
```

Three decisions carry the design:

**A 2D prior, not a 3D one.** This is what makes the project fit one 16 GB GPU:
~7 GB and a day, versus 16–40 GB and a week, and it removes a 3D autoencoder
from the critical path. 3D coherence comes from the physics — rays cross slices,
so the projector couples them — plus an explicit z-direction TV term. Published
practice (DiffusionMBIR, CVPR 2023).

**A real gradient on the data term.** `Projector.bp` is the exact adjoint of
`Projector.fp`, verified by the adjoint gate, so `A^T(A x − p)` is the true
gradient of `½‖A x − p‖²`. Without that the projection loss is a constant and
the regulariser becomes the entire objective — which converges very stably to
smooth mush.

**A two-scale ROI.** Because `A` is linear, `mu = mu_bg + mu_roi` splits exactly:
freeze a coarse background, subtract its contribution from the measurements, and
refine a fine 1 mm box against the residual. That is how 1 mm nodule resolution
fits in under 4 GB, with no approximation.

## Two acquisition tracks

| | Track A — sparse-view CT | Track B — limited-angle DTS |
|---|---|---|
| Views / arc | 8–64 over 180° | 15–25 over 30–40° |
| Posedness | mildly ill-posed | ~150° missing wedge |
| Claim you may make | **absolute** volumetry with error bars | **relative** change detection + uncertainty |

Run Track A first, always. If the pipeline cannot do 32 views over 180° it will
not do 15 over 30, and diagnosing that on the easy case is far cheaper.

## Evaluation

Paired, per-patient reconstruction — so paired metrics, not FID/MMD (which
measure distributional similarity for *unconditional* generation).

- **Global**: PSNR, 3D SSIM, HU-MAE, all **inside the lung mask** (whole-image
  PSNR is dominated by air and flatters every method equally).
- **Nodule**: Dice and absolute percent volume error, computed inside a dilated
  bounding box with connected-component selection — never a whole-volume threshold.
- **`scripts/mdvc.py`**: minimum detectable volume change vs view count and arc.
  This answers the Volume Doubling Time objective directly. LIDC has no follow-up
  scans, so growth is synthesised on the ground truth and re-simulated.
- **`scripts/hallucination.py`**: insert / erase / present conditions →
  detection sensitivity and the **false-positive nodule rate**.
- **`scripts/ablate_roi.py`**: objective O5, Gaussian vs voxel at matched compute.

Baselines a learned prior must beat: FBP, SIRT, SIRT-TV, CGLS.

## Running on Kaggle

Built for it: pure-PyTorch projector, so nothing beyond Kaggle's preinstalled
stack is needed, and no internet.

- **12-hour session cap** → `--max-hours 11` stops cleanly and checkpoints;
  rerunning the same command resumes from `last.pt`. Metrics append to `log.csv`.
- **Paths** are auto-detected: `/kaggle/input` (read-only) and `/kaggle/working`.
  Override with `--data` / `--out`.
- **No wandb** — logging is CSV + matplotlib on disk.
- Full LIDC-IDRI (~124 GB) is not feasible there; use the phantom route, or a
  Kaggle-hosted LUNA/LIDC subset via `prepare_lidc.py --source dir`.

See [notebooks/kaggle_pipeline.ipynb](notebooks/kaggle_pipeline.ipynb).

## Layout

```
src/medlift3d/
  geometry.py     Grid: shape + spacing + origin, and assert_matches       [G1]
  units.py        mu in mm^-1, unbounded; one HU_CLIP; net normalisation   [G3]
  projector.py    differentiable fp, exact adjoint bp, parallel + DTS      [G2]
  phantom.py      synthetic chest phantoms, nodule grow/erase
  prior2d.py      2D UNet noise estimator
  diffusion.py    DDPM + DDIM; config frozen into every checkpoint
  solver.py       prior ⇄ data consistency + z-TV, K-sample posterior
  roi.py          exact two-scale decomposition
  gaussians.py    anisotropic Gaussian field, softplus amplitude
  metrics.py      lung-restricted global + bounded per-nodule metrics      [G4]
  datasets.py     cases with their frames, patient-level splits
  export.py       NIfTI, uncertainty, DICOM SEG/SR, triplanar PNG
  baselines/      fbp, sirt, sirt_tv, cgls
scripts/          run_gates · make_phantoms · prepare_lidc · train_prior ·
                  reconstruct · evaluate · mdvc · hallucination · ablate_roi
configs/          geometry · prior · solver · roi   (geometry.yaml is the single
                  source of truth for acquisition)
tests/            the nine gate suites
docs/             REVIEW · CODE_AUDIT · PLAN · IMPLEMENTATION
```

## Known limitations

Stated plainly, because each one bounds a claim:

- **Simulated projections.** Training and testing on monochromatic simulated
  data is a domain gap. `configs/geometry.yaml` adds Poisson noise; beam
  hardening and scatter are not modelled. Closing this needs real DTS data.
- **Juxtavascular nodules.** Nodules fused to a vessel of the same density
  remain ambiguous for threshold-based volumetry. Vessel-aware segmentation is
  out of scope.
- **Track B absolute volumetry.** A 150° missing wedge is expected to make
  absolute volume error large. That boundary is a result to report, not a defect
  to hide.
- **Phantom realism.** Synthetic phantoms exercise the pipeline; they do not
  substitute for LIDC-IDRI when reporting clinical numbers.
