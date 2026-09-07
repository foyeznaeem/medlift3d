# MedLift-3D — Revised Idea & Implementation Plan (single-GPU feasible)

Companion to `docs/REVIEW.md`. This plan keeps the theme, the title, the clinical
motivation and most of Ch. 1–2 of the report. It changes the *engineering* so the thing
can actually be built by three people on one consumer GPU in a trimester, and so the
claims it makes are ones the experiments can support.

Design budget assumed throughout: **one 12 GB GPU** (RTX 3060/4070 class, or a Colab/Kaggle
T4 16 GB), with occasional access to a second. Every number below is sized for that.

---

## 1. Revised thesis statement

> Under limited-angle and sparse-view chest acquisition, a learned anatomical prior can
> recover volumetric lung structure well enough for **nodule volumetry and growth
> detection**, provided (a) every reconstruction is hard-constrained to agree with the
> measured line integrals, and (b) every voxel carries a calibrated uncertainty so that
> prior-invented structure is visibly distinguishable from measured structure.
>
> MedLift-3D delivers this, and quantifies **where it stops working** — the sparsity and
> angular-coverage threshold below which absolute volumetry is no longer trustworthy.

Two things changed relative to the report. First, the deliverable is no longer "CT-quality
volumes" but "volumetry with stated error bounds plus an honest failure boundary" —
which is both achievable and more useful clinically. Second, uncertainty and hallucination
auditing move from an unimplemented promise in the abstract to first-class contributions.
That is the defensible novelty (see `REVIEW.md` A7); the architecture alone is not.

## 2. Revised objectives (replaces report §1.3)

1. **O1 — Physics-consistent sparse-view lifting.** Build a differentiable forward
   projector for parallel/cone-beam and limited-angle DTS geometry, validated by an adjoint
   test, and use it to hard-constrain reconstruction to the measured projections.
2. **O2 — Low-memory anatomical prior.** Train a chest-CT diffusion prior on a single
   12 GB GPU, and demonstrate it recovers structure that filtered back-projection and
   TV-regularized iterative reconstruction cannot at the same view count.
3. **O3 — Clinically meaningful metrics.** Evaluate on nodule Dice, absolute percent volume
   error, and Bland–Altman agreement against CT-derived volume — not only PSNR/SSIM — and
   report the **minimum detectable volume change** curve as a function of view count and
   angular range.
4. **O4 — Calibrated uncertainty and a hallucination audit.** Produce per-voxel uncertainty
   from posterior sampling, verify its calibration, and quantify invented- and
   erased-nodule rates via controlled insertion/removal experiments.
5. **O5 — Efficient explicit refinement (with an honest ablation).** Implement additive
   X-ray Gaussian splatting for ROI refinement and measure whether it beats voxel-space
   refinement at equal compute and equal wall-clock. Report the result either way.

O4 and O5 are what an examiner will remember. O5 is deliberately falsifiable.

## 3. Two acquisition tracks

Fixes `REVIEW.md` A4. Both are simulated from the same CT ground truth, so this costs
almost nothing extra.

| | Track A — sparse-view CT (**primary**) | Track B — limited-angle DTS (**hard case**) |
|---|---|---|
| Views | N ∈ {8, 16, 32, 64} | 15, 25 |
| Arc | 180° | 30°, 40° |
| Posedness | mildly ill-posed | severe missing wedge (~150°) |
| Comparable to | every method in report Table 2.1 | DOLCE (ICCV'23) |
| Claim you may make | **absolute** nodule volumetry with error bars | **relative** change detection + uncertainty |

Track A is what gets you a working system and publishable numbers. Track B is the report's
original setting and is where the interesting failure analysis lives. Run A first —
it de-risks everything, because if the pipeline cannot do 64 views over 180° it will
certainly not do 15 over 30°.

## 4. Revised architecture

Four stages. Stage 3 is the only mandatory novel machinery; Stage 4 is the ablation.

```
 measured projections p  (V × H × W)
            │
   ┌────────▼─────────┐
   │ S1  Initializer  │  FBP/FDK + SART-TV  →  coarse volume  (256³ @ 1.5mm)
   └────────┬─────────┘
   ┌────────▼─────────┐
   │ S2  Prior        │  2D axial slice diffusion (or 3D latent diffusion, §5)
   └────────┬─────────┘
   ┌────────▼─────────┐
   │ S3  DC solver    │  alternate: prior denoise step ⇄ data-consistency step (A, Aᵀ)
   │                  │  + z-direction TV coupling      → posterior samples ×K
   └────────┬─────────┘
   ┌────────▼─────────┐        ← mean = reconstruction, std = uncertainty map
   │ S4  ROI refine   │  additive X-ray Gaussian splatting, 128³ @ 1.0mm ROI  [ablated]
   └────────┬─────────┘
            ▼
  volume + uncertainty + nodule mask  →  NIfTI, DICOM SEG, DICOM SR
```

### 4.1 The key GPU decision: use a 2D prior, not a 3D one

This is the single change that makes the project fit on one card, and it is not a
compromise — it is published, proven practice for exactly this problem (DiffusionMBIR,
CVPR 2023).

Train the diffusion prior on **2D axial slices** (256×256). Enforce 3D coherence not
through a 3D network but through a **total-variation coupling along z** inside the
data-consistency step. The projector A already couples slices physically, because rays
cross slices; the z-TV term suppresses the inter-slice flicker that a purely 2D prior
would otherwise introduce.

Why this wins on your hardware:

| | 3D latent diffusion (report's plan) | 2D slice diffusion (recommended) |
|---|---|---|
| Prior training VRAM | 16–40 GB (needs the 3D AE first) | **~7 GB** at batch 8, AMP |
| Prior training time | ~1–2 weeks, if the AE gate passes | **~24–36 h** on one 12 GB card |
| Components on critical path | 3D AE **+** 3D diffusion | 2D diffusion only |
| Training data | ~400 volumes | ~400 volumes × ~220 slices ≈ **90k images** |
| Failure modes | AE destroys nodules (`REVIEW.md` A6); latent scale bug (B4) | inter-slice flicker — mitigated by z-TV |

The 3D autoencoder — your single hardest component, and the one that endangers the
clinical objective — leaves the critical path entirely. Keep it as an *upgrade* (§5.1),
gated on the fidelity test, not as a prerequisite.

### 4.2 Stage 3: the data-consistency solver

Alternate a prior step and a measurement step. With `A` the projector, `p` the measured
projections, `x` the volume:

```
for t = T … 1:
    x ← prior_denoise_step(x, t)                       # 2D UNet, applied slicewise
    for j in 1..M:                                     # M = 2–3 CG / gradient steps
        x ← x − η ( Aᵀ(A x − p) + λ_z ∇·TV_z(x) )
    x ← clamp(x, 0, μ_max)                             # attenuation is non-negative
```

Notes that matter in practice:
- `Aᵀ` is back-projection. Never form `A` as a matrix.
- Use **DDIM with 30–50 steps**, not 1000 (`REVIEW.md` C2). Wall clock per case drops to
  seconds.
- Run this K = 8 times with different noise seeds. **Mean → reconstruction. Per-voxel
  std → the uncertainty map** the abstract promised. K=8 at 256³ float16 is ~270 MB — free.
- Slicewise application of the 2D prior means peak VRAM is set by the *projector*, not the
  network.

### 4.3 Stage 4: additive X-ray Gaussian splatting (the corrected physics)

Fixes `REVIEW.md` A3. Represent attenuation as a sum of Gaussians:

```
μ(x) = Σ_i α_i · G_i(x),     G_i(x) = exp( −½ (x−µ_i)ᵀ Σ_i⁻¹ (x−µ_i) ),     Σ_i = R_i S_i S_iᵀ R_iᵀ
```

The line integral of a Gaussian along a ray with unit direction `n` is analytic. For a ray
through point `x₀`:

```
∫ G_i(x₀ + t n) dt  =  sqrt( 2π / (nᵀ Σ_i⁻¹ n) ) · exp( −½ q_i )
```

where `q_i` is the residual quadratic form of `(x₀ − µ_i)` in the plane orthogonal to `n`.
So the rendered projection is a **plain sum** over primitives:

```
p̂_v(u) = Σ_i α_i · sqrt( 2π / (n_vᵀ Σ_i⁻¹ n_v) ) · exp( −½ q_i,v(u) )
```

Three consequences, all favourable:

1. **No depth sorting and no `Π(1−α_j)` transmittance.** Order-independent. The rasterizer
   is *simpler* than vanilla 3DGS and there is no occlusion bias.
2. **No spherical harmonics.** One scalar `α_i ≥ 0` per primitive (unbounded — it is an
   attenuation coefficient, not an opacity). ~11 params/primitive vs ~59 for RGB 3DGS.
3. The `sqrt(2π / nᵀΣ⁻¹n)` amplitude is the path-length-dependent factor that stock 3DGS
   omits. Dropping it is precisely the integration bias R²-Gaussian corrects.

**Implement in pure PyTorch, no CUDA.** Project centres with the camera matrix, compute the
2D covariance `Σ_2D = J Σ Jᵀ`, compute per-primitive 3σ bounding boxes, `scatter_add_` the
contributions onto the detector. Autograd handles the backward exactly. Target ~150 lines.
This deletes the "custom CUDA kernels" task from Table 3.1 (`REVIEW.md` B10).

Regularize the **primitives**, never a rasterized dense volume (`REVIEW.md` B6):

```
L = ‖p̂ − p‖² / (V·H·W)  +  λ_s Σ‖exp(s_i)‖₁  +  λ_a Σ ReLU(−α_i)  +  λ_iso Σ (σ_max/σ_min − 1)⁺
```

`λ_iso` is the important one for Track B: it directly penalizes the z-elongation that the
missing wedge produces.

### 4.4 The two-scale ROI trick (how 1 mm nodule resolution fits in 12 GB)

You cannot afford 512³ @ 1 mm. You do not need it. Nodules are ≤ 30 mm, and DTS reading
workflow already localizes them in 2D.

Because `A` is **linear**, split the volume exactly:

```
μ  =  μ_bg  (coarse, 256³ @ 1.5 mm, from Stage 3, then frozen)
   +  μ_roi (fine,   128³ @ 1.0 mm, zero outside the ROI box)

residual   r  =  p − A μ_bg
refine μ_roi   against r, using only rays that intersect the ROI box
```

No approximation is involved — the background's contribution to every line integral is
subtracted exactly. Stage 4 then optimizes ~100–200k Gaussians inside a 128³ box, which is
comfortable in **under 4 GB**. This ROI decomposition is a legitimate small contribution in
its own right (region-of-interest tomography applied to a generative pipeline) and is worth
its own subsection in the report.

## 5. Optional upgrades (only if Stage 1–4 works and time remains)

**5.1 3D latent diffusion** — keeps your existing PatchVolumeAE/BiFlowNet code alive.
Preconditions, in order:
1. Reduce compression to **4× per axis, C = 4** (≈16× volumetric, not 64×).
2. Add KL (β ≈ 1e-6) and compute an explicit latent scale factor (`REVIEW.md` B4).
3. **Fidelity gate:** encode→decode the ground truth and measure nodule Dice + absolute %
   volume error on the round trip alone. **Require APE ≤ 5% and Dice ≥ 0.85.** If it fails,
   the compression is wrong and no downstream work can repair it. Do not proceed on hope.
4. Only then: 3D UNet on 64³×4 latents — roughly the cost of a 2D UNet on 512², so batch 1–2
   with gradient checkpointing and AMP fits ~12–16 GB.

**5.2 ControlNet** — genuinely useful for *one* thing: adapting a single trained prior across
view counts and geometries without retraining the base (`REVIEW.md` C10 — say it that way).
Not for MRI (`REVIEW.md` B7). Lowest priority; drop without regret.

## 6. Data plan

Fixes `REVIEW.md` B8.

- **Dataset: LIDC-IDRI** (1018 scans, CC BY 3.0, DICOM), read via `pylidc`. It is the only
  choice that gives **voxel-level nodule contours** — four independent readers per nodule.
  GT nodule mask = **≥50% reader consensus**. If you also want LUNA25, use it strictly as an
  extra source of *unlabelled* volumes for prior training, and state its terms.
- **Quality filter:** keep scans with slice thickness ≤ 1.5 mm. Expect ~350–450 usable.
- **Split, patient-level and disjoint:** ~300 train / 50 val / 50 test. **The prior must
  never see a test patient** — otherwise hallucination is indistinguishable from
  reconstruction and the entire safety argument collapses (`REVIEW.md` B9). Persist the
  split as a committed CSV, seeded, never regenerated.
- **Canonical grid, fixed once and asserted in code:** RAS orientation, 256×256×256 @
  1.5 mm isotropic whole-chest; 128×128×128 @ 1.0 mm ROI. HU clip to [−1000, 400], then map
  to linear attenuation (not to [0,1] — the projector needs physical μ).
- **DRR simulation:** prefer **DiffDRR** (differentiable, pip-installable, Siddon
  ray-casting) or **DeepDRR** (models scatter and beam hardening). ASTRA or LEAP for the
  voxel-domain projector/back-projector. Verify that "ChestXSim" (report §4.1.5) is a real,
  citable, obtainable package before relying on it; if it is internal, describe it fully
  or switch.
- **Realism, in order of value:** Poisson photon noise → polychromatic beam hardening →
  scatter. Do the first two; the third only if time allows. Note in Limitations that
  training and testing on simulated projections is a domain gap, and that closing it needs
  real DTS data.

## 7. Evaluation protocol

Fixes `REVIEW.md` B5 and B9. This section is where the marks are — budget real time for it.

### 7.1 Metrics
- **Global fidelity:** PSNR, 3D SSIM, HU-MAE **within the lung mask** (whole-image PSNR is
  dominated by air and flatters everyone).
- **Nodule (the clinical objective):** Dice vs consensus mask; **absolute percent volume
  error (APE)**; Bland–Altman plot vs CT-derived volume; error stratified by nodule
  diameter (<6, 6–10, 10–20, >20 mm) and by type (solid / part-solid / GGO).
- **Growth sensitivity → `minimum detectable volume change`.** LIDC has no follow-up, so
  construct it: dilate a GT nodule by k% volume for k ∈ {5, 10, 20, 30, 50}, re-simulate
  projections, reconstruct, and test whether the measured change is separable from the
  method's own noise floor. **Plot MDVC vs view count vs angular range.** This is the single
  most valuable figure in the thesis — it answers the VDT objective directly, costs only
  compute, and needs no longitudinal data.
- **Uncertainty calibration:** does the K-sample interval achieve nominal coverage? Report
  a reliability diagram and expected calibration error.
- **Efficiency:** seconds/case and peak VRAM, reported honestly per stage.

### 7.2 Hallucination audit (the safety contribution)
1. **Insertion:** add a synthetic nodule to the GT, re-simulate, reconstruct. Does it
   survive? → detection sensitivity vs diameter.
2. **Removal:** inpaint the nodule out of the GT, re-simulate, reconstruct. Does the prior
   invent one back? → **false-positive nodule rate**. Given the 96% LDCT false-positive rate
   in report §1.1, this is the metric a clinician will actually care about.
3. **Out-of-distribution:** reconstruct a case with pathology absent from training (large
   effusion, consolidation, post-surgical anatomy) and show the uncertainty map lights up.

### 7.3 Baselines (all five, non-negotiable)
FBP/FDK · SART + TV (ASTRA) · supervised 3D U-Net (or X2CT-GAN-style) · an INR baseline
(NAF/NeRP) · **R²-Gaussian without a prior** — the last isolates the contribution of the
prior, which is the whole thesis.

### 7.4 Ablations
no prior (S1+S3 only) · no data-consistency (prior only — expect it to look beautiful and be
wrong, which is the point) · no z-TV · **no Stage 4 ROI refinement** (settles O5) ·
K ∈ {1, 4, 8, 16} · λ_iso on/off for Track B.

## 8. VRAM and time budget (one 12 GB GPU)

| Task | Peak VRAM | Wall clock | Notes |
|---|---|---|---|
| Preprocess 400 scans | — | 4–6 h (CPU) | parallelize over cores |
| Simulate DRRs, both tracks | ~3 GB | 6–10 h | cache to `.npz`; run overnight |
| **2D prior training** | **~7 GB** | **24–36 h** | 256², batch 8, AMP, ~60M params |
| Stage 3, one case, K=8 | ~5 GB | 20–60 s | 40 DDIM steps × 2 CG steps |
| Stage 4 ROI refine | ~4 GB | 1–3 min | 150k Gaussians, 3000 iters |
| Full test set (50 cases × 6 configs) | — | 6–10 h | overnight |
| 3D latent diffusion (§5.1, optional) | 12–16 GB | 4–7 days | only if the gate passes |

Total critical-path GPU time: **under one week**, versus a 3D-first plan that does not fit
in 12 GB at all. Fit AMP + gradient checkpointing from day one, not as a rescue.

## 9. Fourteen-week schedule

Mapped to the report's Table 3.1 roles, with the CUDA task removed and the freed time moved
into evaluation.

| Wk | Foyez (Systems) | Jubair (AI/Modeling) | Motasim (Data/Eval) | Gate |
|---|---|---|---|---|
| 1 | canonical grid + affine round-trip test | literature: DOLCE, DiffusionMBIR, R²-Gaussian, X-Gaussian | LIDC via `pylidc`, split CSV | `REVIEW.md` B2 bug closed |
| 2 | projector `A`/`Aᵀ` + **adjoint test** | 2D UNet + DDIM scaffold | DRR simulation, Track A | `⟨Ax,y⟩=⟨Aᵀy,x⟩` passes |
| 3 | FBP/FDK + SART-TV baselines | start prior training | nodule mask pipeline, consensus | **baseline numbers exist** |
| 4 | eval harness, metric library | prior training continues | Track B DRRs | metrics runnable end-to-end |
| 5 | Stage 3 solver (prior ⇄ DC) | monitor/retrain prior | 3D U-Net baseline | prior converged |
| 6 | z-TV coupling, K-sample loop | tune λ_z, DDIM steps, guidance | INR baseline | **first PSNR/Dice on Track A** |
| 7 | ROI decomposition (§4.4) | uncertainty calibration | Bland–Altman, stratified errors | ROI residual verified exact |
| 8 | additive splatter, forward | splatter regularizers | MDVC experiment, Track A | splatter matches `A` on a phantom |
| 9 | splatter backward + refine loop | λ_iso for Track B | MDVC, Track B | **O5 ablation answered** |
| 10 | R²-Gaussian-no-prior baseline | hallucination insert/remove | OOD cases, radiologist review | **false-positive nodule rate** |
| 11 | full ablation sweep | write up method chapter | all figures + tables | results frozen |
| 12 | NIfTI + **DICOM SEG/SR** export | write up results chapter | Gradio/Streamlit + `niivue` viewer | `REVIEW.md` C11 closed |
| 13 | reproducibility: seeds, `requirements.txt`, README | full report revision | UI polish, demo video | one-command repro |
| 14 | buffer | defense prep | buffer | — |

Two hard rules. **Week 6 is the go/no-go**: if there is no PSNR and Dice number for Track A
by then, cut Stage 4 and §5 entirely and ship a complete, well-evaluated
prior + data-consistency system — that is a good FYDP. And **week 11 freezes results**; do
not tune after that.

## 10. Repository layout

```
medlift3d/
  configs/            geometry.yaml, prior.yaml, solver.yaml, splat.yaml
  data/
    prepare_lidc.py       DICOM → canonical RAS 256³ @1.5mm + consensus masks
    simulate_drr.py       Track A/B projections → .npz
    splits.csv            committed, seeded, never regenerated
  medlift3d/
    geometry.py           grid, affine, LPS↔RAS  (owns the B2 bug fix)
    projector.py          A, Aᵀ as torch.autograd.Function  (+ adjoint test)
    prior2d.py            2D UNet + DDIM
    solver.py             Stage 3: prior ⇄ DC + z-TV, K-sample posterior
    roi.py                two-scale decomposition (§4.4)
    splat.py              additive X-ray Gaussian splatting (§4.3)
    metrics.py            PSNR/SSIM/HU-MAE, nodule Dice/APE, MDVC, calibration
    export.py             NIfTI + DICOM SEG + DICOM SR
  baselines/            fbp.py sart_tv.py unet3d.py inr.py r2gaussian.py
  experiments/          run_track_a.py run_track_b.py ablations.py hallucination.py
  tests/                test_adjoint.py test_geometry.py test_roi_exact.py test_splat_vs_projector.py
  app/                  gradio viewer
```

Write `tests/` in week 1–2, not at the end. `test_adjoint.py` and `test_roi_exact.py` each
catch a class of bug that would otherwise silently corrupt every number in Chapter 4.

## 11. Risk register

| Risk | Trigger | Fallback |
|---|---|---|
| 2D prior gives inter-slice flicker | visible z-striping | raise λ_z; go 2.5D (condition on 3 adjacent slices) |
| Track B volumetry never usable | APE > 25% at 15 views/30° | **that is the finding** — report the boundary, lead with Track A |
| Prior erases small nodules | Dice collapses below 6 mm | add nodule-conditioned guidance; restrict claims to ≥6 mm and say so |
| Stage 4 no better than voxel refine | O5 ablation negative | report the negative result; 3DGS stays as the fast viewer only |
| DRR domain gap | reviewer challenges simulation realism | add Poisson + beam hardening; name it in Limitations |
| GPU unavailable | — | 128³ @ 3 mm pilot; every component is resolution-parametric by config |

## 12. Concrete edits to the report text

1. §1.3 — replace all four objectives with §2 above (`REVIEW.md` A1).
2. Abstract — rewrite from the method: no NIF, no "absent volumetric supervision", and keep
   the uncertainty claim only because §4.2/§7 now delivers it (`REVIEW.md` A5).
3. §3.2.3 — add Track A/B and justify the view counts and angular ranges (`REVIEW.md` A4).
4. §3.2.4 — restate 3DGS as **ROI refinement with additive X-ray integration**, with the
   line-integral derivation of §4.3 (`REVIEW.md` A2, A3).
5. New §3.2.5 — the two-scale ROI decomposition (§4.4).
6. §2.2/2.3/Table 2.2 — add DOLCE, DiffusionMBIR, R²-Gaussian, X-Gaussian, X2CT-GAN, and
   restate novelty as *task + uncertainty + failure boundary* (`REVIEW.md` A7).
7. NFR-01 — 512³ → 256³ @ 1.5 mm + 128³ @ 1.0 mm ROI (`REVIEW.md` B3).
8. Objective 4 / NFR-04 — delete MRI generalization; one sentence in Future Work
   (`REVIEW.md` B7).
9. §1.6 and Ch. 4 — FID/MMD → the §7.1 metric set (`REVIEW.md` B5).
10. Ch. 4 — reframe as "pipeline bring-up and feasibility", resolve the contradiction with
    §6.1–6.2, delete §4.2.3, fix §4.3.1 (`REVIEW.md` C1–C3, C9).
11. Eq. 1.1 / FR-04 — add the `(x − µ_i)` centring; unify the `proj_loss` normalization
    (`REVIEW.md` C4, C5).
12. Table 3.1 — drop "custom CUDA kernels"; add evaluation harness and uncertainty
    calibration (`REVIEW.md` B10).
13. §5.1.1/§5.1.3 — drop the HIPAA/GDPR overclaim; add DICOM SEG + SR as a real deliverable
    (`REVIEW.md` C11).
14. Table 5.5 — fix the accumulated cash flow and restate break-even; move revenue to
    Year 2+ (`REVIEW.md` C6, C7).
15. Table 5.6 — get one radiologist to review ~20 reconstructions, claim P6, and reuse it as
    the qualitative evaluation §4.3.2 currently lacks (`REVIEW.md` C12).
16. Full language pass (`REVIEW.md` C13).
