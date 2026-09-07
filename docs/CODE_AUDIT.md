# MedLift-3D — Audit of the FYDP-1 Codebase

Source: `medlift3d-prev-codebase/` (31 files, 26 Python modules + one CUDA kernel).
Provenance: adapted from **3DGR-CT** (Gaussian model + CUDA kernels), **MedSyn** (UNet3D →
BiFlowNet), **ChestXSim** (ASTRA-based DTS simulation).

**Bottom line.** The pipeline is well-organized, the third-party integration work is real,
and roughly 60% of the code is salvageable. But **the system as written cannot perform
patient-specific reconstruction**, for four independent and individually fatal reasons.
All four are silent — nothing crashes, the loss curves look plausible, and wandb reports
convergence. This is precisely why Chapter 4 of the report contains loss numbers and not a
single PSNR, SSIM, or Dice value: there was nothing to report.

---

## The four showstoppers

### S1. Stage 3 has no gradient path from the projection loss — TV is the entire objective

`utils/ct_projector.py` calls itself a *"Differentiable wrapper for ChestXSim's DTS
projector"*. It is not differentiable:

```python
vol_cupy   = cp.from_dlpack(torch.utils.dlpack.to_dlpack(vol_xyz))
projs_cupy = self.opt.project(vol_cupy, self.vx_xyz)      # ASTRA — no autograd
projs      = torch.utils.dlpack.from_dlpack(projs_cupy.toDlpack())
```

DLPack shares **memory**, not the autograd graph. The returned tensor has no `grad_fn`,
and there is no `torch.autograd.Function` anywhere wrapping this call. So in
`train_medlift3d.py`:

```python
train_projs = projector.forward_project(train_output...)   # grad_fn = None
proj_loss   = F.mse_loss(train_projs, projections[0])      # requires_grad = False
tv_loss     = tv_regularization(train_output.unsqueeze(1)) # differentiable ✓
loss        = proj_loss + config['tv_weight'] * tv_loss
loss.backward()
```

`loss.backward()` **does not raise**, because `tv_loss` carries a valid graph through
`grid_sample → compute_intensity`. Gradients flow — but *only from the total-variation
regularizer*. `proj_loss` is a constant added to the objective.

The consequence is that for 8000 iterations the optimizer minimizes total variation with
**zero data consistency**. TV minimization alone drives the field toward piecewise-constant
mush; the measurements are never consulted. The loop is not "physics-consistent
refinement" — it is an anatomy eraser. Densification still runs (because `_xyz.grad` is
non-`None`, courtesy of TV), the progress bar shows a falling loss, and nothing warns you.

Verify it yourself in one line:

```python
p = projector.forward_project(v); assert p.requires_grad and p.grad_fn is not None
```

### S2. The diffusion sampler interprets the model's output as the wrong quantity

`train_diffusion.py` trains with:

```python
GaussianDiffusion(..., loss_type='l2', objective='pred_noise')
```

`train_medlift3d.py` reconstructs the sampler with **no such arguments**:

```python
diffusion = GaussianDiffusion(denoise_fn=denoise_fn, image_size=None,
                              timesteps=config['diffusion_timesteps'])
```

and `GaussianDiffusion.__init__` defaults to `objective='pred_x0', loss_type='l1'`. So
`p_mean_variance` takes the network's **ε-prediction** and uses it directly as `x_start`.
The reverse process is fed noise where it expects clean latents, at every one of 1000
steps. The sampled latent is garbage regardless of how well Stage 2 trained.

Compounding it, the same function then does:

```python
x_start.clamp_(-1., 1.)
```

The AE's latents come from an unregularized `nn.Linear(128, 64)` — arbitrary scale, no KL,
no VQ, no normalization constant. Clamping them to [−1, 1] corrupts them even when the
objective is right. (This is `REVIEW.md` B4, now confirmed in code with an extra hard
clamp on top.)

### S3. The conditioning pathway destroys all spatial information

Two `AdaptiveAvgPool` layers sit exactly where the patient-specific signal has to pass.

**`models/xray_encoder.py`** — the projections enter here and leave as global descriptors:

```python
nn.Conv2d(256, feature_dim, 3, stride=2, padding=1),
nn.AdaptiveAvgPool2d(1),   # 536×536 detector  →  1×1
nn.Flatten()               # one 512-vector per view
```

Each X-ray becomes **one 512-dim vector**. The conditioning signal handed to the diffusion
model is 15 global numbers-bags that encode roughly "a chest with about this much total
attenuation" and nothing whatsoever about *where* anything is. You cannot localize a
nodule from a spatially-pooled descriptor. And `InterPatchFlow` — the only place this
context enters the network — is applied **once, at the UNet bottleneck**, at the coarsest
resolution.

**`models/patch_volume_ae.py`** — the same mistake on the volume side:

```python
nn.Conv3d(64, 128, 3, stride=2, padding=1),
nn.AdaptiveAvgPool3d(1),   # 4×4×4  →  1×1×1
nn.Flatten(),
nn.Linear(128, latent_dim) # 4096 voxels → 64 numbers
```

Each 16³ patch (4096 voxels) is **globally average-pooled** before the linear map. This is
much worse than a strided-conv VAE at the same nominal 64× ratio: the pool annihilates all
spatial arrangement *within* the patch, so the decoder must invent 16³ of structure from a
spatially-averaged summary. A 5 mm nodule occupies a fraction of one patch. It cannot
survive. (`REVIEW.md` A6, confirmed and worse than estimated.)

Net effect: the prior is architecturally incapable of being patient-specific. It is a
generic chest generator.

### S4. The projection loss compares incommensurate quantities

`chestxsim_settings/luna25_dts.json` simulates the measured projections with:

```json
"unit_conversion": {"units": "density", ...},
"Physics_effect": {"I0": 1e5, "voltage": 120, "poly_flag": true,
                   "apply_flood_correction": true, "log": true}
```

i.e. polychromatic, flood-corrected, log-transformed projections of a **density** field.

The rendered projections come from `forward_project()` applied to the Gaussian volume,
whose intensities are `torch.sigmoid(...)` ∈ (0,1) — normalized units, no density
conversion, no `I0`, no polychromatic model, no flood correction, no log. `F.mse_loss`
then compares raw monochromatic ASTRA line integrals of a [0,1] field against
log-normalized polychromatic measurements. Different units, different scale, different
nonlinearity. Even with S1 repaired, this loss is not a data-consistency term.

**Rule going forward: the forward model in the loss must be bit-for-bit the forward model
that generated the data.** Simulate with the operator you optimize through, or not at all.

---

## Additional confirmed defects

**Geometry and units**

- **`reco_vx: [1.25, 5.00, 1.25]`** in the FDK config is the root cause of `REVIEW.md` B2 —
  the report's `324×65×94` FDK shape. The 5 mm axis is DTS's poor depth direction. The FBP
  volume therefore lives on a **different grid with different spacing** from the GT volume.
  `create_from_fbp` then normalizes coordinates into a unit cube by *that* grid's shape,
  while `train_medlift3d.py` renders on a grid built from `gt_volume.shape`. The anatomy is
  silently stretched anisotropically. Nothing tracks an affine anywhere past preprocessing.
- **`3_package_npz.py` normalizes the FBP volume as if it were HU:**
  ```python
  fbp_volume = np.clip(fbp_volume, -1000, 400); fbp_volume = (fbp_volume + 1000)/1400.0
  ```
  ChestXSim's FDK output is in **density/attenuation** units (~0–2), not HU. Clipping those
  to [−1000, 400] is a no-op, and the affine map sends them to ≈ 0.714–0.715. **Every `fbp`
  array in every `.npz` is very likely a near-constant field.** Which means
  `create_from_fbp`'s gradient-magnitude sampling sees ~0 everywhere, `topk` returns
  arbitrary voxels, and the Gaussians are initialized at essentially random positions. High
  confidence from the config plus the code; confirm by printing
  `data['fbp'].min(), .max(), .std()` on any packaged case.
- **HU clipping is inconsistent between stages.** `1_mha_to_nifti.py` clips to
  **[−1000, 3000]** and that NIfTI is what ChestXSim simulates from. `3_package_npz.py`
  clips the GT to **[−1000, 400]**. So the projections contain bone attenuation up to
  3000 HU that the training target saturates away. A perfect reconstruction could not match
  the measurements.
- **`spacing = np.abs(np.diag(affine)[:3])`** is only correct for axis-aligned volumes. Use
  `np.linalg.norm(affine[:3,:3], axis=0)`.
- **Four different view counts** across the repo: `luna25_dts.json` → `60`,
  `configs/medlift3d.yaml` → `15`, `DTSProjector.__init__` → `5`, `XRayEncoder.__init__` →
  `5`. The report says 15. Geometry is configured in two unlinked places.
- **The low-res stage is geometrically inconsistent.** `grid[:, ::2, ::2, ::2, :]` renders a
  half-resolution volume, which is then projected with **full-resolution `vx_xyz`** — wrong
  physical scale — and compared against `projections[0][:, ::2, ::2]`. Decimating detector
  pixels is not the same operation as projecting a decimated volume.
  `KAGGLE_DEBUG_LOG.md` §11 shows this was introduced as a shape-error workaround.

**Evaluation — why no clinical metric was ever produced**

- **`utils/metrics.py` computes Dice over the whole volume.**
  ```python
  pred_binary = (pred > 0.5).astype(float)          # ALL soft tissue + bone, 512×512×184
  intersection = (pred_binary * nodule_mask).sum()
  metrics['dice'] = 2*intersection / (pred_binary.sum() + nodule_mask.sum() + 1e-8)
  ```
  There is no crop to the nodule ROI and no connected-component selection. Dice is
  structurally forced to ~1e-4. `volume_error_pct` divides the same whole-volume count by
  the nodule volume, so it reports errors on the order of 10⁵ %.
- **`nodule_mask` is never written.** `evaluate.py` reads `data['nodule_mask']`, but
  `3_package_npz.py` never stores that key and `4_extract_nodules.py` only saves *intensity
  patches* as separate `.npy` files. So `nodule_mask` is always `None` and the clinical
  branch never executes. This is the mechanical reason the report has no Dice or volume
  error.
- **`4_extract_nodules.py` has an unresolved axis bug**, acknowledged in its own comments
  ("You may need to swap v_x, v_y, v_z"). The guard `if volume.shape == (512, 512, Z)`
  evaluates — given `Z, X, Y = volume.shape` — to `shape == (512,512,512)`, false for real
  data, so it always takes the `else` branch and indexes a `[X,Y,Z]` array as `[Z,X,Y]`:
  dim 0 (size 512) gets a z-index, dim 2 (size ~184) gets a y-index up to 511. Patches are
  out-of-range slices.
- **`psnr(gt, pred, data_range=1.0)`** assumes pred ∈ [0,1], but the Gaussian field is an
  *additive* sum of sigmoid-weighted primitives and routinely exceeds 1.
- **`ssim(gt, pred, channel_axis=0)`** computes per-axial-slice 2D SSIM and averages it.
  That is 2.5D SSIM; report it as such or compute true 3D SSIM.
- **No train/val/test split exists anywhere.** Both trainers use `shuffle=True` over the
  whole directory and select "best" on **training** loss (so the report's "Epoch 196: best
  model" is a training-loss minimum). Stage 3 then reconstructs the same cases the prior
  memorized. Total leakage — `REVIEW.md` B9, confirmed.

**Correctness and robustness**

- **`train_medlift3d.py` never imports `os`** but calls `os.makedirs("results", ...)` in the
  every-2000-iteration checkpoint block. Guaranteed `NameError` at iteration 2000 — i.e. the
  safety checkpointing that `KAGGLE_DEBUG_LOG.md` §13 was written to provide crashes the run.
- **The diffusion warm start is dead code.** The comment says *"Start diffusion from FBP
  latent encoding to speed up sampling"*, but:
  ```python
  fbp_latent     = ae.encode(fbp_padded)
  latent_refined = diffusion.p_sample_loop(shape=fbp_latent.shape, context=xray_context)
  ```
  only `.shape` is used. Sampling starts from pure noise; the encode is wasted compute.
- **Silent zero placeholders.** `3_package_npz.py` writes
  `np.zeros((15,512,512))` for missing projections and `np.zeros_like(volume)` for missing
  FDK, then prints a warning and continues. Cases with no simulation output become
  all-zero training samples. Raise instead.
- **`affine` is returned as `None`** by `luna25_loader` when absent; the default collate
  will crash on `None`.
- **No DDIM and no classifier-free guidance.** Only the full 1000-step `p_sample_loop`, and
  `context` is never dropped during training, so conditioning strength is not tunable.
  (`REVIEW.md` C2.)
- **`accelerator.prepare(ae, ...)`** wraps a fully frozen module in DDP; the
  `ae.module.encode if hasattr(...)` dance in `train_diffusion.py` is the symptom.
- **The AE's checkpointing trick is fragile by construction.** `chunk.requires_grad_(True)`
  works only because the input volume does not require grad; `KAGGLE_DEBUG_LOG.md` §3
  records discovering this the hard way. It is correct for an autoencoder but will break
  silently if the AE is ever placed downstream of another module.
- **Per-forward-pass patch loop.** A 384×384×96 crop at `p=16` is 3456 patches, chunked at
  32 → **108 sequential encoder invocations per forward pass**. This, not attention, is why
  Stage 1 was slow.

**Report-vs-code mismatches (integrity-relevant)**

- **`BiFlowNet`'s "novelty" is not what the report describes.** `IntraPatchFlow` is plain
  self-attention over the whole feature map; `InterPatchFlow` is plain cross-attention to
  the conditioning vector. Neither is patch-local or patch-to-patch. Report §1.3/§3.1.3
  claim "intra-patch flow for finer-grained local information and inter-patch flow for
  consistent overall structure" — the code implements a standard conditional 3D UNet with
  attention. The names describe an intent that was never built. This directly undercuts the
  Table 5.7 A3 "Innovation" claim.
- **Report Eq. 4.2 states ε-prediction with L2.** The default configuration is
  `pred_x0` + **L1**, and Stage 2 trains `pred_noise` + L2. The reported "loss 1.0438 →
  0.5130" is therefore an L2 ε-loss whose absolute value is uninterpretable (`REVIEW.md` C1),
  and it was minimized on training data with no validation.
- **`3DGS` provides no memory advantage in this implementation.** `grid_sample` evaluates
  every Gaussian at every point of a full `[1, Z, X, Y, 3]` grid via `discretize_grid.cu`
  and materializes the dense volume **every iteration** (the `grid` tensor alone is
  ~578 MB at 512×512×184). 3DGR-CT is a *Gaussian-parameterized voxel grid*, not a splat
  rasterizer. Report §3.2.4 and §4.1.3 justify 3DGS on "efficient memory usage" and
  "real-time projection computation" — untrue as built. (This *corrects* `REVIEW.md` A3 on
  one point: the physics is **not** alpha-blending. Gaussians are summed additively onto a
  grid and then line-integrated by ASTRA, which is the right *kind* of forward model. The
  problem is cost and the broken gradient, not occlusion.)

---

## Salvage assessment

| Component | Verdict | Action |
|---|---|---|
| `models/gaussian_model.py` + `utils/gs_utils/` (3DGR-CT) | **Keep — most valuable asset** | Working forward *and* backward CUDA kernels with correct additive semantics. Hard to rewrite. Repurpose for Stage 4 ROI refinement at 128³ where the dense-grid cost is affordable. |
| `preprocess/1_mha_to_nifti.py` | Keep, ~80% | LPS→RAS negation is correct. Fix spacing extraction; unify the HU clip with stage 3. |
| `preprocess/2_run_chestxsim.py` | Keep the hard-won parts | The ChestXSim monkey-patches (`VolumeExtender`, `PhysicsEffect`, `ct_vx`/`ct_dim` injection) represent real debugging value. Preserve them verbatim. |
| `models/biflownet.py` | Keep as reference, ~60% | Competent conditional UNet. `ResnetBlock`, `SinusoidalPosEmb`, and the attention blocks port to 2D almost unchanged. Rename the two "Flow" modules honestly. |
| `models/diffusion_prior.py` | Keep, ~70% | Standard DDPM with a correct cosine schedule. Add DDIM, fix the objective plumbing, delete the `clamp_`, add CFG dropout. |
| `utils/luna25_loader.py`, `export_nifti.py`, `visualize.py` | Keep, trivial | Add split handling; fix the `affine=None` collate crash. |
| `KAGGLE_DEBUG_LOG.md` | **Keep verbatim** | 13 documented environment fixes. Fold into the report's testing section in place of §4.2.3. |
| `utils/ct_projector.py` | **Rewrite** | Must become a `torch.autograd.Function` (S1). See `IMPLEMENTATION.md` §2. |
| `models/xray_encoder.py` | **Rewrite** | Delete `AdaptiveAvgPool2d(1)`; keep spatial feature maps (S3). |
| `models/patch_volume_ae.py` | **Drop from critical path** | `AdaptiveAvgPool3d(1)` is unfixable at this ratio (S3). The recommended 2D-prior path removes the need entirely. |
| `utils/metrics.py` | **Rewrite** | Whole-volume Dice is meaningless. See `IMPLEMENTATION.md` §5. |
| `preprocess/3_package_npz.py` | **Rewrite** | FBP unit bug, silent zero placeholders, no nodule masks, no affine discipline. |
| `preprocess/4_extract_nodules.py` | **Delete** | Axis bug, and centroid CSVs cannot yield segmentation masks. Replace with LIDC-IDRI + `pylidc` contours (`REVIEW.md` B8). |
| `train_medlift3d.py` | **Rewrite** | Becomes the data-consistency solver. |
| `medlift3d.zip` (61 MB) | Delete | A snapshot of the same tree; do not commit binaries. |
| `models/__pycache__/*.pyc` | Delete | Add `.gitignore`. |

## Why the report's Chapter 4 looks the way it does

Every gap in that chapter is explained by a specific defect above, which is worth knowing
because it tells you what to re-run rather than what to re-word:

| Report observation | Cause |
|---|---|
| No PSNR/SSIM anywhere | S1 + S2 — there was no meaningful reconstruction to measure |
| No Dice or volume error | `nodule_mask` never written; whole-volume Dice would read ~1e-4 anyway |
| "Stable convergence, no memory failures" as the headline result | The only live gradient was TV, which converges very stably to mush |
| FDK shape `324×65×94` | `reco_vx: [1.25, 5.00, 1.25]`, no affine tracking |
| Loss 1.0438 → 0.5130 uninterpretable | ε-MSE on training data, no validation split, objective mismatched at inference |
| `NameError: checkpoint` promoted to a results subsection | Symptom of S1: with no real metric to report, the debug log became the result |
