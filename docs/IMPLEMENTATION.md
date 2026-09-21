# MedLift-3D — Detailed Implementation Plan (FYDP-2)

Supersedes the build sections of `docs/PLAN.md` now that the FYDP-1 code has been audited
(`docs/CODE_AUDIT.md`). Read `CODE_AUDIT.md` first — this document assumes the four
showstoppers S1–S4 as given.

Budget: **one ~12 GB GPU**, 14 weeks, three people.

---

## 0. Order of work, and why

The audit changes the priority order. Four things must be true before *any* model training
is worth a single GPU-hour, because each one silently invalidates every downstream number:

```
W1  G1  geometry + affine discipline        (one canonical grid, asserted)
W2  G2  differentiable projector            (adjoint test passes)          ← fixes S1
W2  G3  simulate-with-what-you-optimize     (units test passes)            ← fixes S4
W3  G4  metrics that mean something         (nodule Dice on a phantom)
────────────────────────────────────────────────────────────────────────────
W3+     only now: baselines, then the prior
```

Gates G1–G4 are ~2.5 weeks of unglamorous work that the previous attempt skipped. That
skip is the whole reason eight months produced no reconstruction metric. Do not skip it
again. Every gate is a `pytest` file, not a judgement call.

## 1. Gate G1 — one canonical grid, and an affine that is never dropped

The single deepest defect in the old code is that no array ever knew where it was in
space. Fix it with a type, not a convention.

`medlift3d/geometry.py`:

```python
from dataclasses import dataclass
import numpy as np, torch

@dataclass(frozen=True)
class Grid:
    """A voxel grid with an explicit physical frame. RAS, mm."""
    shape:   tuple          # (nz, ny, nx)  -- index order, always
    spacing: tuple          # (sz, sy, sx)  mm
    origin:  tuple          # (oz, oy, ox)  mm, RAS

    @property
    def affine(self) -> np.ndarray:
        A = np.eye(4)
        A[:3, :3] = np.diag(self.spacing)
        A[:3,  3] = self.origin
        return A

    def assert_matches(self, other: "Grid", tol=1e-4):
        assert self.shape == other.shape, f"shape {self.shape} != {other.shape}"
        assert np.allclose(self.affine, other.affine, atol=tol), "affine mismatch"

# The two grids the whole project uses. Nothing else is permitted.
CHEST = Grid(shape=(256, 256, 256), spacing=(1.5, 1.5, 1.5), origin=(0., 0., 0.))
ROI   = Grid(shape=(128, 128, 128), spacing=(1.0, 1.0, 1.0), origin=(0., 0., 0.))
```

Rules, enforced by tests:

1. Index order is **always** `(z, y, x)`. The old code called the same array `[X,Y,Z]` in
   preprocessing and `[Z,X,Y]` in training; the padding arithmetic happened to survive it,
   but `4_extract_nodules.py` did not. Pick one order and never rename.
2. Every `.npz` stores `shape`, `spacing`, `origin`. Every loader reconstructs a `Grid` and
   calls `assert_matches`.
3. Compute spacing as `np.linalg.norm(affine[:3,:3], axis=0)` — never `np.diag`.
4. **The GT and the reconstruction live on the same grid.** No more `324×65×94`. The
   anisotropic `reco_vx: [1.25, 5.00, 1.25]` FDK output is used *only* as a
   visualization/initialization aid, resampled onto `CHEST` with its affine, never compared
   to anything directly.

`tests/test_geometry.py`: round-trip a real case DICOM → `CHEST` → NIfTI → reload, and
assert shape and affine equality plus HU-range preservation.

## 2. Gate G2 — the differentiable projector (fixes S1)

This is the most important 60 lines in the project. The Radon transform is **linear**, so
its Jacobian is its adjoint: the backward pass of forward-projection is back-projection.
Wrap ASTRA/ChestXSim once, correctly, and every downstream stage becomes differentiable.

`medlift3d/projector.py`:

```python
import torch

class _Project(torch.autograd.Function):
    """y = A x   with   dL/dx = Aᵀ (dL/dy).  A is linear, so this is exact."""

    @staticmethod
    def forward(ctx, volume, op):
        ctx.op = op
        ctx.vol_shape = volume.shape
        with torch.no_grad():
            return op.fp(volume)            # torch->cupy->ASTRA->cupy->torch

    @staticmethod
    def backward(ctx, grad_projs):
        with torch.no_grad():
            grad_vol = ctx.op.bp(grad_projs.contiguous())   # the ADJOINT, not FDK
        return grad_vol.reshape(ctx.vol_shape), None


def project(volume, op):
    return _Project.apply(volume.contiguous(), op)
```

Three things the old code got wrong here and that the wrapper must respect:

- **`bp` must be the plain adjoint back-projection, not FDK.** FDK applies a ramp filter;
  it is a *reconstruction* operator, not `Aᵀ`. Using it as the backward pass gives wrong
  gradients that still look like they are converging. ASTRA exposes both — use `BP`, not
  `FDK`/`FBP`.
- Keep the DLPack zero-copy transfer; it is a genuinely good idea (and the old
  `ct_projector.py` got the DLPack mechanics right). The bug was never DLPack — it was the
  absence of an `autograd.Function` around it.
- Fix the permutation once, inside `op.fp`/`op.bp`, and assert the output shape. The
  `[H,W,V]` vs `[V,H,W]` confusion is `KAGGLE_DEBUG_LOG.md` §11.

**`tests/test_adjoint.py` — the gate. Nothing proceeds until this passes:**

```python
def test_adjoint(op, grid):
    x = torch.rand(grid.shape,       device='cuda')
    y = torch.rand(op.proj_shape,    device='cuda')
    lhs = (op.fp(x) * y).sum()          # <A x, y>
    rhs = (x * op.bp(y)).sum()          # <x, Aᵀ y>
    assert torch.allclose(lhs, rhs, rtol=1e-3), f"{lhs.item()} vs {rhs.item()}"

def test_gradient_flows(op, grid):
    x = torch.rand(grid.shape, device='cuda', requires_grad=True)
    p = project(x, op)
    assert p.requires_grad and p.grad_fn is not None      # would have caught S1
    p.pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
```

Also add `tests/test_no_silent_zero_grad.py`: assert that a projection-only loss (TV weight
set to 0) produces non-zero gradients on the Gaussian parameters. That single assertion is
what eight months of the previous attempt lacked.

## 3. Gate G3 — simulate with the operator you optimize through (fixes S4)

The old pipeline simulated measurements with a polychromatic, flood-corrected,
log-transformed density model and then optimized against monochromatic line integrals of a
`sigmoid`-bounded field. Two incompatible physics models on either side of one MSE.

Resolve it by defining **one** attenuation representation and **two** matched operators:

```
μ(x)  :  linear attenuation coefficient, mm⁻¹, non-negative, UNBOUNDED
         from HU:  μ = μ_water * (1 + HU/1000),  μ_water ≈ 0.02 mm⁻¹ @ 70 keV

A_mono :  p = ∫ μ dl                      -- ASTRA line integral. Differentiable (§2).
A_poly :  the ChestXSim chain             -- realism. NOT differentiable, NOT in the loss.
```

Then pick one of two configurations, and state which in the report:

- **Track A/B main experiments — monochromatic.** Simulate with `A_mono` (+ Poisson noise
  on the pre-log intensity). The loss uses exactly `A_mono`. Data consistency is then
  exact, and this is what essentially every sparse-view CT paper you cite does.
- **Domain-gap experiment — polychromatic.** Simulate with `A_poly` (ChestXSim, `poly_flag:
  true`), reconstruct with `A_mono`. Report the resulting degradation as a *finding* about
  model mismatch and beam hardening. This is a genuinely interesting extra result and it
  reuses all of `preprocess/2_run_chestxsim.py`.

Consequential fixes:

- **Remove every bounded activation on attenuation.** `PatchVolumeAutoencoder`'s final
  `nn.Sigmoid()` and `GaussianModel.intensity_activation = torch.sigmoid` both cap μ at 1.0.
  Use `softplus` or `exp`, or predict normalized μ and carry the scale explicitly.
- **Unify the HU window.** One constant, `HU_CLIP = (-1000, 400)`, imported by preprocessing
  *and* simulation. The old `[-1000, 3000]`-then-`[-1000, 400]` split meant the projections
  encoded bone the target had saturated away.
- **Never re-normalize an FBP/FDK volume with the GT's HU affine map.** That is the bug that
  turned every stored `fbp` array into a ≈0.714 constant. Convert units explicitly, or store
  the raw FDK output plus its unit tag and convert at load time.

`tests/test_units.py`: project a water cylinder of known radius; assert
`p_max ≈ μ_water · 2R` to within 2%. This catches every unit error in one line.

## 4. Stage plan

Keeping the four-stage structure from `PLAN.md §4`, now with concrete salvage decisions.

### S1 — Initializer
FDK via ASTRA, then 30 iterations of SART-TV, resampled onto `CHEST`. Also the first
baseline. Reuses `preprocess/2_run_chestxsim.py`'s ASTRA setup.

### S2 — Prior: 2D axial slice diffusion
Per `PLAN.md §4.1`, and the audit strengthens the case considerably — the 3D path's two
hardest components are the two that are broken beyond repair at this compression ratio.

- Port `ResnetBlock`, `SinusoidalPosEmb`, and the attention blocks from `biflownet.py`
  to 2D. This is a mechanical `Conv3d → Conv2d` change; the code is sound.
- Train on 256×256 axial slices of μ. ~7 GB at batch 8 with AMP, 24–36 h.
- **Delete `x_start.clamp_(-1., 1.)`.** Normalize μ to roughly unit variance with a fixed,
  stored constant instead (compute it once over the training set, commit it to the config —
  Stable Diffusion's `0.18215` plays this role).
- Pass `objective` and `loss_type` **explicitly at every construction site**, and add
  `tests/test_diffusion_config.py` asserting the training and sampling configs are
  identical. That test is S2's permanent fix.
- Add DDIM (30–50 steps) and CFG (drop `context` with p=0.1 during training).

**Keep `patch_volume_ae.py` out of the critical path.** If time remains, `PLAN.md §5.1`
gives the upgrade path — but only behind the fidelity gate, and the `AdaptiveAvgPool3d(1)`
must go regardless (replace with a strided conv to 4×4×4 and a `Conv3d` to 4 channels).

### S3 — Data-consistency solver
Per `PLAN.md §4.2`, now buildable because §2 exists. `K = 8` posterior samples → mean is
the reconstruction, per-voxel std is the uncertainty map. This is what replaces the old
`train_medlift3d.py`.

### S4 — ROI refinement, reusing 3DGR-CT
The audit's most useful positive finding: `gaussian_model.py` + `discretize_grid.cu` have a
**working, tested forward and backward** with correct additive semantics. Do not rewrite
them, and do not write the pure-PyTorch splatter proposed in `PLAN.md §4.3` — it is no
longer needed.

What must change:

- Run it on the **`ROI` grid (128³)**, not the full chest. `grid_sample` materializes a
  dense volume every iteration (~578 MB of grid tensor alone at 512×512×184); at 128³ that
  is ~8 MB and entirely affordable. This is what makes 3DGR-CT usable on 12 GB.
- Feed it the two-scale residual from `PLAN.md §4.4`: freeze the coarse `CHEST`
  reconstruction, subtract `A μ_bg` from the measurements, refine `μ_roi` against
  `r = p − A μ_bg`. Exact, because `A` is linear.
- Replace `intensity_activation = sigmoid` with `softplus` (§3).
- Replace the volume-space `tv_regularization` with primitive-space regularizers
  (`PLAN.md §4.3`): scale L1, non-negativity, and the anisotropy penalty `σ_max/σ_min` that
  fights missing-wedge elongation.
- **Correct the report's justification.** 3DGR-CT is a *Gaussian-parameterized voxel grid*,
  not a splat rasterizer; it buys you a compact, adaptive, edge-aware parameterization —
  not "real-time rendering" or "efficient memory usage" (`CODE_AUDIT.md`, last row). Say
  that plainly and the O5 ablation becomes an honest test of a real question: does an
  adaptive Gaussian parameterization beat plain voxels at equal compute?
- Fix the low-res stage properly: build a genuine coarse `Grid` with correct spacing and a
  matching projector, rather than decimating detector pixels.

## 5. Gate G4 — metrics that mean something

Rewrite `utils/metrics.py`. The old whole-volume Dice cannot be repaired by tuning.

```python
def nodule_metrics(pred_mu, gt_mu, nodule_mask, grid, hu_threshold=-300.):
    """Nodule metrics are computed INSIDE a dilated bounding box, never whole-volume."""
    box = dilate_bbox(bbox_of(nodule_mask), margin_mm=10., grid=grid)
    p, g, m = pred_mu[box], gt_mu[box], nodule_mask[box]

    thr  = hu_to_mu(hu_threshold)
    pred_seg = largest_connected_component(p > thr)      # one nodule, not all tissue

    dice = 2*(pred_seg & m).sum() / (pred_seg.sum() + m.sum() + 1e-8)
    vox  = float(np.prod(grid.spacing))
    ape  = abs(pred_seg.sum() - m.sum()) * vox / (m.sum()*vox + 1e-8) * 100
    return {"nodule_dice": dice, "volume_ape_pct": ape,
            "pred_vol_mm3": pred_seg.sum()*vox, "gt_vol_mm3": m.sum()*vox}
```

Plus, per `PLAN.md §7.1`: PSNR and 3D SSIM computed **inside the lung mask** with
`data_range` taken from the actual μ range (never hardcoded `1.0` — the additive Gaussian
field exceeds 1); true 3D SSIM or an explicit "per-slice 2D SSIM" label.

**Nodule masks come from LIDC-IDRI via `pylidc`, at ≥50% reader consensus** — written into
the `.npz` as `nodule_mask`, which the old pipeline never did. Delete
`4_extract_nodules.py`; centroid CSVs cannot produce segmentation ground truth.

`tests/test_metrics.py`: on a synthetic sphere of known radius, assert
`volume_ape_pct < 2` and `nodule_dice > 0.9`. Had this existed, the whole-volume Dice bug
would have been caught in week 1.

## 6. Data pipeline

Rewrite `3_package_npz.py`; keep `1_mha_to_nifti.py` and `2_run_chestxsim.py`.

- **Switch to LIDC-IDRI** (`REVIEW.md` B8) — 1018 scans, CC BY 3.0, four-reader XML
  contours. It is the only source that supports the clinical metrics. Keep LUNA25 only as
  extra *unlabelled* volumes for prior training.
- Filter to slice thickness ≤ 1.5 mm (~350–450 usable scans). Resample to `CHEST`.
- Commit `data/splits.csv`: patient-level disjoint, ~300/50/50, seeded, never regenerated.
  **The prior never sees a test patient.** Add `tests/test_no_leakage.py` asserting the
  three ID sets are disjoint.
- **Raise on missing simulation output.** The `np.zeros((15,512,512))` and
  `np.zeros_like(volume)` placeholders silently poisoned the training set.
- Store per case: `mu` (float32, `CHEST`), `nodule_mask` (uint8), `projections_A`,
  `projections_B`, `shape`/`spacing`/`origin`, `lidc_id`, `n_views`, `arc_deg`.
- **Geometry lives in exactly one file.** The old repo had four different view counts
  (`luna25_dts.json`: 60, `medlift3d.yaml`: 15, `DTSProjector`: 5, `XRayEncoder`: 5).
  Put geometry in `configs/geometry.yaml`, have the simulator and the projector both read
  it, and assert `projections.shape[0] == cfg.n_views` at load.
- Select `best` on **validation** loss, not training loss.

## 7. Repository layout

As `PLAN.md §10`, with salvage noted:

```
medlift3d/
  configs/       geometry.yaml  prior.yaml  solver.yaml  splat.yaml
  data/          prepare_lidc.py  simulate.py  splits.csv
  medlift3d/
    geometry.py    NEW   §1 — Grid, affine discipline
    projector.py   NEW   §2 — autograd.Function     ← fixes S1
    units.py       NEW   §3 — HU ↔ mu, one HU_CLIP  ← fixes S4
    prior2d.py     port of biflownet.py to 2D
    diffusion.py   from diffusion_prior.py: +DDIM +CFG, −clamp, explicit objective
    solver.py      NEW   §4-S3 — replaces train_medlift3d.py
    roi.py         NEW   two-scale decomposition
    gaussian_model.py    KEEP from 3DGR-CT (sigmoid→softplus, ROI grid)
    gs_utils/            KEEP verbatim — working CUDA fwd+bwd
    metrics.py     REWRITE §5
    export.py      from export_nifti.py: +DICOM SEG/SR
  baselines/     fdk.py  sart_tv.py  unet3d.py  inr.py  gaussian_no_prior.py
  experiments/   track_a.py  track_b.py  ablations.py  hallucination.py  mdvc.py
  tests/         test_geometry.py  test_adjoint.py  test_units.py  test_metrics.py
                 test_no_leakage.py  test_diffusion_config.py  test_roi_exact.py
                 test_no_silent_zero_grad.py
  app/           gradio viewer
  docs/          REVIEW.md  CODE_AUDIT.md  PLAN.md  IMPLEMENTATION.md  KAGGLE_DEBUG_LOG.md
```

Add `.gitignore` for `__pycache__`, `*.pyc`, `*.zip`, `checkpoints/`, `results/`, `data/`.
Do not commit `medlift3d.zip`.

## 8. Revised 14-week schedule

Weeks 1–3 are now gate work. This is the change from `PLAN.md §9`, and it is the change
that matters.

| Wk | Foyez (Systems) | Jubair (AI/Modeling) | Motasim (Data/Eval) | Gate |
|---|---|---|---|---|
| 1 | `geometry.py`, `units.py` | read DOLCE, DiffusionMBIR, R²-Gaussian, X-Gaussian | LIDC + `pylidc`, `splits.csv` | **G1** + `test_no_leakage` |
| 2 | **`projector.py` + adjoint test** | port `biflownet.py` → `prior2d.py` | `prepare_lidc.py` → `CHEST` | **G2** ← fixes S1 |
| 3 | `simulate.py` on `A_mono`, water-cylinder test | `diffusion.py`: DDIM, CFG, config test | **rewrite `metrics.py`** + sphere test | **G3 + G4** ← fixes S4 |
| 4 | FDK + SART-TV baselines | start prior training | eval harness, `nodule_mask` in npz | baseline numbers exist |
| 5 | `solver.py` (prior ⇄ DC) | prior training continues | 3D U-Net baseline | prior converged |
| 6 | z-TV coupling, K-sample loop | tune λ_z, DDIM steps, guidance | INR baseline, Bland–Altman | **first Track A PSNR + Dice — GO/NO-GO** |
| 7 | `roi.py` + `test_roi_exact` | uncertainty calibration | stratified errors by size/type | ROI residual exact |
| 8 | wire 3DGR-CT onto `ROI` grid | softplus + primitive regularizers | MDVC curve, Track A | splatter matches `A_mono` |
| 9 | Track B + λ_iso | Track B tuning | MDVC curve, Track B | **O5 ablation answered** |
| 10 | `gaussian_no_prior` baseline | insertion/removal audit | OOD cases, radiologist review | **false-positive nodule rate** |
| 11 | full ablation sweep | method chapter | all figures and tables | **results frozen** |
| 12 | DICOM SEG/SR export | results chapter | Gradio + `niivue` viewer | `REVIEW.md` C11 closed |
| 13 | seeds, `requirements.txt`, README | full report revision | demo video | one-command repro |
| 14 | buffer | defense prep | buffer | — |

Week 6 remains the go/no-go: no Track A PSNR **and** nodule Dice by then → cut S4 and all
of `PLAN.md §5`, and ship a well-evaluated prior + data-consistency system. That is a good
FYDP. What is not a good FYDP is another eight months of loss curves.

## 9. Environment corrections

- `astra-toolbox` is **not** on PyPI as a working wheel — install via
  `conda install -c astra-toolbox astra-toolbox`. `requirements.txt` listing it as a pip
  dep will fail. `environment.yml` gets this right; delete the pip line.
- `monai==0.8.0` is from 2021 and is not used anywhere in the code. Drop it or upgrade.
- `odl` is listed in both dependency files but never imported. Drop it.
- `numpy<2` + `scipy<1.12` are real ASTRA/CuPy constraints — keep, and comment why.
- Pin actual verified versions and commit `requirements.txt` + `environment.yml` from a
  working env. The report's "PyTorch v2.4.1 / ASTRA-toolbox v2.4.1" (`REVIEW.md` C8) does
  not match `environment.yml`'s `pytorch==2.1.0`; resolve before the report claims either.
- Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to the README rather than setting
  it inside training scripts — it must be set before `import torch` to take effect, which
  the old scripts do correctly but fragilely.

## 10. Report corrections this audit adds

Beyond `PLAN.md §12`:

17. **§4.1.3 / §3.2.4 — remove the "3DGS is memory-efficient / real-time" justification.**
    The implementation materializes a dense volume every iteration. Rejustify Gaussians as
    an adaptive edge-aware parameterization and let the O5 ablation decide.
18. **§3.1.3 / §1.3 — rename or rebuild `IntraPatchFlow` / `InterPatchFlow`.** As written
    they are ordinary self- and cross-attention. Either implement genuinely patch-local and
    patch-to-patch attention, or describe them accurately and drop the dual-flow novelty
    claim (Table 5.7, A3).
19. **§4.2.2 Eq. 4.2 — state the objective actually used** (ε-prediction, L2) and note that
    Stage 2's reported loss was training-set only, selected without a validation split.
20. **§4.1.5 — correct the ASTRA claim.** State explicitly that forward projection is
    wrapped in a `torch.autograd.Function` whose backward is the adjoint back-projection.
    The FYDP-1 code did not do this, which is `CODE_AUDIT.md` S1.
21. **Add a "Verification and Testing" section** (replacing §4.2.3's `NameError` anecdote)
    listing the nine gate tests in §7. This is the strongest available evidence of
    engineering rigour, and it maps directly onto Table 5.6's P3 (Depth of Analysis).
22. **Acknowledgements / §2.2 — cite 3DGR-CT, MedSyn, and ChestXSim in the report itself,
    not only in the code README.** `gaussian_model.py`, `gs_utils/`, and the BiFlowNet
    backbone are adapted from them. This is currently the largest attribution gap between
    the code and the report.
