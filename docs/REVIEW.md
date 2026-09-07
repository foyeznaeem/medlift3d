# MedLift-3D — Technical Review of FYDP-1 Report

Reviewed: `FYDP_Report.pdf` (64 pp., 11 Jun 2026). Findings are ordered by how much
damage they do to the project, not by page order.

**Verdict.** The clinical motivation is sound and well-argued (Ch. 1–2 is the strongest
part of the report). The *engineering* idea, as written, will not work: the three stages
of the pipeline do not compose, the rendering physics is wrong for X-ray, the stated
objectives describe a different and much larger project than the method, and the headline
clinical claim is not attainable from the stated acquisition geometry. All of it is
fixable without abandoning the theme — see `docs/PLAN.md`.

---

## A. Idea-level defects (must fix)

### A1. The Objectives section describes a different project than the rest of the report

§1.3 sets four objectives: a Patch-Volume Autoencoder, a BiFlowNet noise estimator
targeting **512×512×512** generation, a **ControlNet** conditioning layer, and
**generalization across head / chest / abdomen / leg and across CT *and MRI***.

Nothing else in the report is about that. §1.4, Ch. 3 and Ch. 4 are about one thing:
lifting sparse chest projections into a 3D lung volume for nodule assessment. §1.3 reads
as a contribution list imported from a general-purpose 3D medical *generation* paper —
multi-organ, multi-modality, ControlNet-adapted, 512³ — which is a 2–3 person-year
project, not an FYDP.

Consequences: (a) your objectives are unfalsifiable within the project, so Ch. 4 can
never claim to have met them; (b) if PatchVolumeAE/BiFlowNet are adopted from an
existing paper, presenting them under "Innovation" (Table 5.7, A3) without citation is
an integrity problem, and an examiner who knows the source will find it.

**Fix:** rewrite §1.3 to the lung-nodule scope (5 objectives in `PLAN.md §2`). If you keep
PatchVolumeAE/BiFlowNet, cite the source paper explicitly and describe your contribution
as *adaptation + physics-consistent conditioning*, not as the architecture itself.

### A2. The three stages do not compose — the pipeline is a voxel → Gaussian → voxel round trip

Read §4.1.1–4.1.4 literally:

1. AE compresses the CT volume into a latent.
2. Diffusion, conditioned on X-rays, produces a **"complete and dense volumetric
   representation of the lung"** (§4.1.2, *Result*).
3. 3DGS then converts that volume into Gaussians, renders them, and optimizes against
   the projections.
4. `export_nifti.py` converts the Gaussians **back to a voxel grid** (§4.3.2).

If step 2 already yields a dense volume, step 3 adds no information — it re-parameterizes
a volume you already have, and step 4 undoes the re-parameterization. Two lossy
resamplings for zero gain. 3DGS earns its keep when you have *no* volume and must
optimize geometry directly from images; you have placed it *after* the volume exists and
*before* converting back to voxels.

This is the single biggest logical hole in the report and it is the first thing a
reviewer will ask about.

**Fix:** 3DGS must either become the *only* representation (no dense volume ever
materialized, no voxel export) or be demoted from "the reconstructor" to a **local
refiner on a small ROI**, with an ablation that proves it beats voxel-space refinement
at equal compute. `PLAN.md` takes the second option and makes the ablation a deliverable
— a negative result there is still a legitimate FYDP finding.

### A3. 3D Gaussian Splatting's rasterizer implements the wrong physics for X-ray

The report says only that "a differentiable rasterizer projects the Gaussian
representation into two-dimensional views" (§3.1.3, FR-05, §4.1.5). The stock 3DGS
rasterizer performs **front-to-back alpha compositing with spherical-harmonic,
view-dependent color**:

```
C(u) = Σ_i c_i(view) α_i Π_{j<i} (1 - α_j)      # occlusion, depth-sorted, order-dependent
```

X-ray formation is a **line integral** (Beer–Lambert, after `-log(I/I₀)`):

```
p(u) = ∫ μ(x) dl                                 # additive, order-independent, no occlusion
```

There is no occlusion in transmission imaging, no view-dependent appearance, and the
`Π(1-α_j)` transmittance term is physically meaningless here. Optimizing a projection
loss through an alpha-blending rasterizer will drive the geometry to the wrong place —
tissue behind dense structures gets systematically suppressed, which is exactly where
nodules hide. This is the "integration bias" that R²-Gaussian (NeurIPS 2024) was written
to fix, and X-Gaussian (ECCV 2024) drops SH for the same reason.

Your own data dictionary shows the confusion: §3.2.2 calls the parameter
`a (intensity) … Density, α ∈ [0,1]`, §3.1.3 calls the same thing "opacity", and §4.1.5
calls it "voxel intensities". It is an **attenuation coefficient** — non-negative and
*unbounded*, not an opacity in [0,1].

**Fix:** additive splatting with the analytic line integral of a 3D Gaussian, derived in
`PLAN.md §4.3`. Bonus: no depth sort is required, so it is simpler *and* faster than
vanilla 3DGS, and the gradients are exact.

### A4. The headline clinical claim is not achievable from 15 views over 30°

§3.2.3 states the setting: **15 projections over 30°**. §1.2.2 sets the clinical bar:
beat the 47–85% volumetric error of 2D assessment and enable reliable Volume Doubling
Time.

These are incompatible. A 30° arc leaves a ~150° missing wedge; depth resolution
degrades to the order of centimetres and reconstructions are systematically elongated
along the beam axis. VDT on a 6–10 mm nodule needs volume accuracy on the order of a few
percent. Clinical chest DTS itself uses roughly 60 projections over 30–40° and is
explicitly *not* used for volumetry — your own §1.2.4 says so, citing the missing-wedge
problem, and then §1.5.2 promises "more authentic calculation of VDT" anyway. You have
argued against your own claim.

Also: no rationale is given anywhere for *why* 15 views, or what dose that corresponds to.

**Fix:** two tracks (`PLAN.md §3`).
- **Track A (primary, defensible):** sparse-view CT — N ∈ {8, 16, 32, 64} over 180°.
  Well-posed enough for absolute volumetry, directly comparable to every paper in your
  Table 2.1, and cheap to run.
- **Track B (the hard case, the honest one):** limited-angle DTS, 15–25 views over 30–40°.
  Change the claim from *absolute volumetry* to **relative/longitudinal change detection
  with reported uncertainty**, and publish the sparsity–accuracy curve that shows where
  absolute volumetry breaks. That curve is a real contribution; an overclaim is not.

### A5. The Abstract contradicts the method in three places

| Abstract says | Method actually does |
|---|---|
| "combines a **Neural Implicit Field (NIF)** … with a latent diffusion prior" | §3.2.3 explicitly **rejects** implicit/NeRF representations and uses explicit 3DGS |
| "anatomically reasonable reconstructions **when only direct volumetric supervision will be absent**" | AE and diffusion are trained on ground-truth CT volumes — fully volumetrically supervised (Eq. 4.1, 4.2) |
| evaluated on "**uncertainty-common error bounds**" | no uncertainty is computed anywhere in Ch. 3 or Ch. 4 |

The abstract also claims MedLift-3D "removes appearance modeling from geometric structure
to decoupling space and appearance consistency" — geometry/appearance decoupling is
meaningless for X-ray, which has a single scalar attenuation channel and no
view-dependent appearance.

**Fix:** rewrite the abstract from the method, not the other way round. Then *implement*
the uncertainty it promises — it is cheap (K posterior samples, voxelwise std) and it is
the strongest safety argument you have.

### A6. 64× latent compression destroys exactly the structures you are measuring

§3.2.4 and §3.2.2: 16×16×16 patches → a 64-dimensional latent. That is 4096× spatial
reduction into 64 channels, i.e. **64× volumetric compression**. For reference, Stable
Diffusion's VAE compresses 8× spatially in 2D with 4 channels ≈ 16×, on natural images
where losing detail is cosmetic.

A 5 mm nodule at 1 mm spacing occupies ~65 voxels — a *fraction of one patch*. At 64×
compression the autoencoder will smooth it into the surrounding parenchyma before the
diffusion model ever sees it. The clinical objective is destroyed in stage 1, and no
amount of downstream refinement recovers it.

**Fix:** treat AE fidelity as a **hard gate** before any diffusion training. Measure
nodule Dice and absolute % volume error on an *encode→decode round trip of the ground
truth* (no reconstruction, no sparsity). If round-trip APE > ~5%, the compression is too
aggressive — full stop. `PLAN.md §5.1` sets 4× spatial / C=4 (≈16× volumetric) as the
starting point, and `PLAN.md §4.1` offers a path that removes the 3D AE from the critical
path entirely.

### A7. The novelty claim in Table 2.2 is false as stated

Table 2.2's bottom row claims MedLift-3D is the first to combine medical reconstruction +
deep learning + generative AI + sparse-view CT + 3D Gaussian Splatting + lung screening,
and §2.3 says "there is no currently available work" doing so. Prior work you have not
cited:

- **R²-Gaussian** (NeurIPS 2024) — radiative Gaussian splatting for tomography; identifies
  and corrects the exact integration bias in A3.
- **X-Gaussian** (ECCV 2024) — Gaussian splatting for sparse-view X-ray; drops SH.
- **DOLCE** (ICCV 2023) — diffusion prior for **limited-angle** CT. This is your direct
  competitor and your closest baseline.
- **DiffusionMBIR** (CVPR 2023) — 2D diffusion prior + z-direction TV for 3D sparse-view
  CT. Also the cheapest known route to your goal (see `PLAN.md §4.1`).
- **X2CT-GAN** (CVPR 2019) — biplanar X-ray → CT; the obvious supervised baseline.

Leaving these out is what makes the gap look empty. Once they are in, the honest and
still-defensible novelty is narrower and better: **the clinical task** — nodule
volumetry and minimum detectable volume change under limited-angle acquisition, **with
calibrated per-voxel uncertainty and an explicit hallucination audit**. Nobody has
published that, and it is achievable at FYDP scale.

---

## B. Serious technical errors

**B1. The tool list mixes two mutually exclusive rendering paths, and ASTRA is not
differentiable.** §4.1.5 lists ASTRA-toolbox for projection *and* 3DGS for
representation. ASTRA operates on voxel grids and has no PyTorch autograd; 3DGS needs a
splat rasterizer. You cannot backprop a projection loss through ASTRA as listed. (The
fix is easy and worth stating in the report: the Radon transform is linear, so wrap
forward-projection in a `torch.autograd.Function` whose backward is ASTRA's
back-projection — the adjoint. But you must say so.) Decide per stage: ASTRA/LEAP/DiffDRR
for voxel stages, your own additive splatter for the Gaussian stage.

**B2. Table 4.1 evidences a live geometry bug.** GT `512×512×184` reconstructs to
"FDK Volume Shape" `324×65×94`. A reconstruction that is to be compared against ground
truth by PSNR/SSIM/Dice **must live on the same grid in the same physical frame**. A
324×65×94 grid is not comparable to anything, and the axis magnitudes suggest a
transposed geometry plus an LPS/RAS mix-up on top of a mis-specified detector/volume
spec. Fix before anything else: define **one** canonical grid (RAS, 256³ @ 1.5 mm),
reconstruct into exactly that grid, and add a unit test asserting
`recon.shape == gt.shape and allclose(recon.affine, gt.affine)`.

**B3. NFR-01 (512³) is unachievable from your data.** Your own Table 4.1 shows z ranging
111–184 slices. LIDC/LUNA chest CT is ~512×512×(100–400) with anisotropic spacing
(0.6–3.0 mm). 512³ would require upsampling z by 3–5×, i.e. inventing the resolution you
claim to validate. Restate NFR-01 as 256×256×256 @ 1.5 mm isotropic whole-chest, plus a
128³ @ 1.0 mm nodule ROI.

**B4. A pure-L2 autoencoder produces a latent unfit for diffusion.** Eq. 4.1 is
unnormalized squared error and nothing else. With no KL or VQ regularizer the latent has
arbitrary, per-dataset scale and unbounded outliers; diffusion's `√ᾱ_t z₀` noise schedule
assumes roughly unit variance. Trained as written, stage 2 will underperform for reasons
that look like a modelling failure but are a normalization bug. Add a small KL term
(β ≈ 1e-6) *and* an explicit latent scale factor (SD uses 0.18215; compute yours from the
training set). Also normalize Eq. 4.1 by voxel count and add a gradient/edge term —
plain L2 alone blurs nodule margins.

**B5. FID and MMD are the wrong metrics.** §1.6 promises FID and MMD "to prove valid
performance compared with state-of-the-art baselines". Both measure *distributional*
similarity for **unconditional** generation. Yours is a **paired, per-patient
reconstruction** task: you have the ground truth for every test case, so use paired
metrics. FID/MMD would only be appropriate for a side experiment benchmarking the
unconditional prior, and 2D-Inception-based FID on CT slices is a known-poor proxy
anyway. Replace with PSNR/SSIM/HU-MAE + nodule Dice/APE (`PLAN.md §7`).

**B6. The TV loss defeats the point of using Gaussians.** Eq. 4.7 computes total
variation on `V_gs` — the dense volume rasterized from the Gaussians. Materializing a
256³ float volume every iteration to regularize it reintroduces the memory cost 3DGS was
chosen to avoid. Regularize the *primitives* instead: scale penalty
`Σ‖exp(s_i)‖₁` (kills needle artifacts), a non-negativity/sparsity penalty on `α`, and an
anisotropy penalty `max(σ)/min(σ)` (directly counteracts the missing-wedge elongation in
A4).

**B7. MRI generalization is physically unsupported.** Objective 4 and NFR-04 claim one
framework covers CT *and* "fast MRI reconstruction" via ControlNet. Your entire
physics-consistency argument rests on a differentiable **line-integral projection**
operator. MRI's forward model is a Fourier/k-space sampling operator — different domain,
complex-valued, different artifacts. The projector, the DTS geometry, and the DRR
simulator all fail to transfer. Drop it from the objectives; at most keep one sentence in
Future Work: "the same latent-prior + data-consistency scheme can be re-instantiated with
a Fourier forward operator."

**B8. Dataset inconsistency, and the wrong ground truth for the clinical task.** §4.2.1
says raw data is **LUNA25** `.mha`; §5.1.1 says source data is **LIDC-IDRI** DICOM under
CC BY. Pick one and be consistent. More importantly: nodule Dice and volume error need
**voxel-level nodule masks**, and challenge sets of the LUNA family generally ship nodule
*centroids* (and, for LUNA25, malignancy labels), not contours. **LIDC-IDRI is the right
choice** — it carries per-radiologist XML contours from four readers, is CC BY 3.0, and is
readable with `pylidc`. Use the ≥50% reader-consensus mask as GT. Also state and check the
redistribution terms for whichever set you use.

**B9. No baselines and no split protocol.** Ch. 4 reports no comparison against any
method, and no patient-level train/val/test split is stated anywhere. Without a split,
the diffusion prior may have seen your test patients — and then hallucination is
indistinguishable from reconstruction, which invalidates the whole safety argument.
Mandate: patient-level disjoint split, the prior never sees a test patient, and a fixed
list of five baselines (`PLAN.md §7.3`).

**B10. "Custom CUDA kernels" is a schedule bomb.** Table 3.1 assigns "development and
optimization of custom CUDA kernels for differentiable rasterization". A correct 3DGS
backward pass in raw CUDA is a multi-month specialist task and it is not what is being
graded. The additive X-ray splatter in `PLAN.md §4.3` is ~150 lines of pure PyTorch,
needs no hand-written kernel, and autograd gives you the exact backward for free. Reassign
that time to evaluation and the uncertainty work, which is where the marks are.

---

## C. Reporting errors (cheap to fix, all visible to an examiner)

- **C1. §4.3.1 reports no results.** Diffusion ε-prediction MSE falling 1.0438 → 0.5130
  says almost nothing: that loss starts near 1.0 by construction and its absolute value is
  not interpretable across models. "8000 iterations with stable convergence and no memory
  allocation failures" is a log line, not a result. Not one reconstruction-quality number
  (PSNR, SSIM, Dice) appears in the entire chapter.
- **C2. §4.3.1 numbers are internally inconsistent.** "1000 denoising timesteps, requiring
  27.0 seconds and achieving approximately 36.37 steps/sec": 1000/27 = 37.0, and
  36.37 × 27 = 982. Also, 1000 DDPM steps is unnecessary — DDIM or DPM-Solver++ gives
  equivalent quality in 25–50 steps, a 20–40× inference speedup you are currently leaving
  on the table.
- **C3. Ch. 4 and Ch. 6 contradict each other.** Ch. 4 presents a trained three-stage
  system with loss curves; §6.1–6.2 say "FYDP-1 is still in the early development phases"
  and that the "main constraint … is the lack of full system implementation". Pick one
  story. (Recommended: Ch. 4 = "pipeline bring-up and feasibility", with the AE gate from
  A6 as its actual result.)
- **C4. Eq. 1.1 / FR-04 are mis-stated.** `G(x) = exp(-½ xᵀΣ⁻¹x)` is centred at the
  origin. It must be `G_i(x) = exp(-½ (x-µ_i)ᵀ Σ_i⁻¹ (x-µ_i))`. `Σ = RSSᵀRᵀ` (§3.2.2) is
  correct — keep it.
- **C5. `proj_loss` is normalized differently in two places.** §3.2.4's table gives
  `1/(VHW) Σ‖·‖²`; Eq. 4.6 gives an unnormalized `Σ_v ‖·‖²`. Fix one definition and use it
  everywhere; it changes the meaning of λ_TV.
- **C6. Table 5.5's accumulated cash flow is arithmetically wrong.** Cash flow
  (−632, 98, 118, 138, 158, 178, 198, 218) accumulates to
  (−632, −534, −416, −278, −120, 58, 256, 474). The table prints
  (−797, −699, −581, −443, −285, −107, 91, 309) — a constant −165 offset with no stated
  opening balance. This also moves the break-even point, so the §5.3 conclusion changes.
- **C7. Revenue model is not credible as written.** 200k revenue in Q1 while the product
  is still being built; "0.2k **sterling pounds** per month" mixes currency with an
  otherwise unlabelled "Thousands"; 100k/quarter of single-user subscriptions at 0.2k/mo
  implies ~167 paying radiologists from day one. Move all revenue to Year 2+, label the
  currency once, and state the subscriber-count assumption.
- **C8. Suspicious version numbers.** PyTorch "v2.4.1" and ASTRA-toolbox "v2.4.1" —
  identical, likely a copy-paste. Verify every version against your actual environment and
  freeze a `requirements.txt`; examiners do check reproducibility claims.
- **C9. §4.2.3 does not belong in a thesis.** A `NameError: name 'checkpoint' is not
  defined` and its one-line import fix is a commit message, not a section. Replace with the
  testing protocol that *should* be there: unit tests for the geometry round trip (B2), the
  projector adjoint (`⟨Ax,y⟩ = ⟨x,Aᵀy⟩`), and the HU normalization.
- **C10. FR-06 misdescribes ControlNet.** "adapted to downstream reconstruction tasks …
  without having to be retrained" — ControlNet *is* a fine-tuning method. It freezes the
  base model and trains a new conditioning branch. Say "without retraining the base
  prior".
- **C11. Compliance section overclaims and under-delivers.** §5.1 claims HIPAA/GDPR
  alignment, but LIDC-IDRI is already public and de-identified — there is no PHI to
  protect, so the claim is unearned. Meanwhile you claim DICOM interoperability and a REST
  API but produce only `.nii.gz`. Swap the fluff for something real and cheap: **export the
  nodule mask as a DICOM SEG plus a DICOM SR with the volume measurement.** That is genuine
  interoperability, worth a paragraph, and about a day of work with `pydicom`/`highdicom`.
- **C12. Table 5.6 leaves P6 (stakeholder involvement) blank** with no explanation, while
  §5.4.2/A2 claims engagement with "domain experts". Either get one radiologist to review
  ~20 reconstructions (also gives you the qualitative evaluation §4.3.2 currently lacks) and
  claim P6, or state plainly why it does not apply.
- **C13. Prose quality.** Several sentences are unparseable and will cost marks
  independently of the technical content — e.g. Abstract: "a generated model Hybridization
  model", "grasp refined Hour texture"; §2.3: "explicit 3D Gaussian **disorders**"; §5.1.3:
  "so insecuring code with (seed = 42)"; §1.5.3 / §3.4 similar. A full language pass is
  needed. Several passages read as machine-translated or paraphrase-tool output; that is a
  risk beyond style.
