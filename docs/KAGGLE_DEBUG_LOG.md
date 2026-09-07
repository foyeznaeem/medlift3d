# MedLift-3D: Kaggle Multi-GPU Debugging & Optimization Log

This document serves as a historical record of the specific Out-of-Memory (OOM) crashes, shape mismatches, and architectural bugs encountered while migrating the MedLift-3D pipeline to the Kaggle Dual-T4 (16GB) GPU environment. It details *why* each problem occurred and exactly *what* code changes were made to solve them.

---

## 1. Stage 0: Projection Settings Update
**File Modified**: `chestxsim_settings/luna25_dts.json`
* **Goal**: Increase the amount of structural information passed to the generative models.
* **Change**: Increased `num_projections` from `5` to `15`. 
* **Impact**: Required downstream configuration files to be updated to expect 15 views instead of 5.

---

## 2. Stage 1: Autoencoder OOM (Memory Fragmentation & Stack Size)
**Files Modified**: `train_autoencoder.py` and `configs/medlift3d.yaml`
* **Problem**: Training the fully convolutional 3D Autoencoder on raw `512x512x168` CT volumes caused an instant Out-Of-Memory (OOM) crash on 16GB GPUs. Additionally, heterogenous patient scan depths caused `torch.stack` to crash when attempting to batch multiple patients together.
* **The Fixes**:
    1. **Accelerate DDP**: Replaced standard PyTorch loop with HuggingFace `accelerate`. This splits independent batch sizes across both Kaggle GPUs, allowing `ae_batch_size: 1` to process two patients simultaneously without `torch.stack` crashes.
    2. **Mixed Precision**: Forced `fp16` mixed precision via `accelerate` to cut model weight size in half.
    3. **Dynamic Spatial Cropping**: Injected a PyTorch bounds-checking loop to randomly crop `384x384x96` spatial volumes from the patients. This slashed the memory footprint by 50% while still forcing the model to learn 3D geometries.
    4. **Memory De-fragmentation**: Injected `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` and disabled `torch.backends.cudnn.benchmark` to prevent the PyTorch CUDA allocator from panicking and crashing when memory fragmented during the backward pass.

---

## 3. Stage 1: Autoencoder Architecture Bugs & Silent Checkpoint Failures
**File Modified**: `models/patch_volume_ae.py`
* **Problem 1 (Logic Error)**: The patch extraction `unfold()` logic was referencing the wrong tensor dimension index, causing an arithmetic crash.
* **Problem 2 (Memory Corruption)**: Flattening the tensor without a contiguous `.permute()` mixed spatial channels together.
* **Problem 3 (Silent Checkpoint Bypass)**: Even though `checkpoint.checkpoint()` was used to save VRAM, the model still OOM'd during the backward pass. PyTorch was silently bypassing the checkpoint mechanism and caching all 7.5GB of activations because the input chunks did not have gradients enabled!
* **The Fixes**:
    1. Fixed the `tensor.unfold(4)` index bug and added a strict `.permute()` re-ordering block.
    2. Forced `chunk.requires_grad_(True)` right before the checkpoint to force PyTorch to respect the memory limits and recalculate the network instead of caching it.
    3. Aggressively lowered the encoder `chunk_size` to 32, crushing the peak memory usage during the backpropagation recalculation.

---

## 4. Stage 2: Diffusion Config Mismatch
**File Modified**: `configs/medlift3d.yaml`
* **Problem**: `AssertionError: Expected 5 views, got 15`. The `XRayEncoder` was hard-coded by the config to expect 5 X-Ray projections, but Stage 0 was now generating 15.
* **The Fix**: 
    1. Updated `num_views: 15` and `dts_nprojs: 15` in the YAML config.

---

## 5. Stage 2: BiFlowNet U-Net Spatial Misalignment
**File Modified**: `train_diffusion.py`
* **Problem**: `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 4 but got size 5`. The `BiFlowNet` is a 3-stage 3D U-Net. For its skip-connections to stitch together perfectly, its input (the Latent Grid) MUST be a perfect multiple of `2³ = 8`. The latent grids extracted from the raw CT volumes were not divisible by 8, causing the network to misalign by 1 pixel internally.
* **The Fix**: 
    1. Added a dynamic `torch.nn.functional.pad` block right before the Autoencoder forward pass. 
    2. The script now calculates exactly how much padding is needed to stretch the original CT volume to the nearest multiple of `128` (since `16 * 8 = 128`). This guarantees the extracted latent grid is always perfectly divisible by 8.
    3. *Bonus*: Fixed a PyTorch `FutureWarning` by adding `weights_only=True` to the `torch.load()` command.

---

## 6. Stage 2: Quadratic Self-Attention VRAM Explosion (4GB per layer)
**Files Modified**: `models/biflownet.py` and `train_diffusion.py`
* **Problem**: `CUDA out of memory. Tried to allocate 4.00 GiB.` The `BiFlowNet` was attempting to execute `IntraPatchFlow` (Self-Attention) on the highest-resolution latent layer (`16x32x32`). This generated a sequence length of `16,384` pixels. Comparing every pixel to every other pixel required a massive `16384 x 16384` matrix, demanding over 4GB of VRAM for a single mathematical operation!
* **The Fix**: 
    1. Standard Diffusion models (like Stable Diffusion) never apply attention to their highest resolutions. Patched `BiFlowNet.__init__` to selectively disable `IntraPatchFlow` on the first two high-resolution layers by checking `if use_attn = ind >= 2`. Attention is now only used in the deep bottleneck where sequence lengths are tiny, instantly curing the 4GB explosion.
    2. Copied the `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` de-fragmentation hack over to `train_diffusion.py` to match the Autoencoder script's resilience.

---

## 7. Stage 2: BiFlowNet Channel Shape Mismatch (Missing `init_conv`)
**File Modified**: `models/biflownet.py`
* **Problem**: `RuntimeError: Expected weight... got weight of shape [128] and input of shape [1, 64, 32, 32, 16]`. A standard U-Net populates a list of "skip connections" (`h`) as it goes down the encoder, and pops them as it goes up the decoder to concatenate features. The final `ResnetBlock` in `BiFlowNet` was mathematically initialized to expect `128` channels (64 from the current layer + 64 from the absolute base skip connection). However, the absolute base skip connection was never saved because whoever wrote the architecture forgot to add the initial `init_conv`! Because `h` was missing its very first item, the decoder completely ran out of skip connections to pop, feeding a `64` channel tensor into a `128` channel layer.
* **The Fix**: 
    1. Added the missing `self.init_conv = nn.Conv3d(...)` to the start of the `BiFlowNet` class.
    2. Correctly applied `x = self.init_conv(x)` in the `forward` pass and populated the initial skip-connection array `h = [x]`.

---

## 8. Stage 3: Gen-3DGS Prior Initialization Shape Mismatch
**File Modified**: `train_medlift3d.py`
* **Problem**: `RuntimeError: Calculated padded input size per channel: (7 x 3 x 3). Kernel size: (4 x 4 x 4). Kernel size can't be greater than actual input size`. The `BiFlowNet` model performs 3-levels of downsampling which strictly requires the spatial dimensions of the latent grid to be perfectly divisible by 8. During Stage 2 diffusion training, a dynamic padding step ensured the input volume dimensions were a multiple of `128` (`patch_size * 8`). However, this padding step was missing in Stage 3, causing the autoencoder to output an incompatible latent grid shape during the reverse diffusion prior generation.
* **The Fix**: 
    1. Added the same dynamic padding logic to `fbp_volume` before passing it to the autoencoder.
    2. Added a dynamic slicing (unpadding) step to restore the decoded `diffusion_volume` back to its exact original dimensions before 3D Gaussian Splatting initialization.

---

## 9. Stage 3: Gaussian Splatting Initialization Shape Mismatch
**Files Modified**: `train_medlift3d.py` and `models/gaussian_model.py` (context)
* **Problem**: `ValueError: not enough values to unpack (expected 5, got 4)`. The `create_from_fbp` function in `gaussian_model.py` expects a 5-dimensional volume tensor with a channel dimension at the end (`[batch_size, D, H, W, Channels]`). However, the `diffusion_volume` passed to it only had 4 dimensions (`[batch_size, Z, X, Y]`).
* **The Fix**: 
    1. Appended `.unsqueeze(-1)` to `diffusion_volume` in `train_medlift3d.py` when passing it to `create_from_fbp()`. This adds a dummy channel dimension, successfully satisfying the 5D structural expectation without altering any mathematical properties of the single-channel intensity data.

---

## 10. Stage 3: DLPack Tensor Permutation Crash
**Files Modified**: `train_medlift3d.py`
* **Problem**: `RuntimeError: permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 4 is not equal to len(dims) = 3`. The `ct_projector.py` wrapper specifically expects a 3D tensor (`[Z, X, Y]`) to safely pass it to the ChestXSim CuPy backend via DLPack. The `train_output` from `gaussian_model.py` possessed a shape of `[1, Z, X, Y, 1]`. By running `.squeeze(-1)`, the dummy channel was removed leaving `[1, Z, X, Y]`, but the leading batch dimension remained, causing the 4D-to-3D DLPack permutation to crash.
* **The Fix**: 
    1. Replaced `train_output.squeeze(-1)` with `train_output.squeeze(0).squeeze(-1)` to safely strip *both* the batch dimension and the dummy channel dimension, perfectly aligning it with the required 3D format for CuPy.

---

## 11. Stage 3: ChestXSim DTS Projector Resolution Mismatch
**Files Modified**: `utils/ct_projector.py` and `train_medlift3d.py`
* **Problem**: `RuntimeError: The size of tensor a (15) must match the size of tensor b (268) at non-singleton dimension 2`. This crash had two massive compounding causes:
    1. **Permutation Bug**: The `forward_project` function in `ct_projector.py` was returning the tensor in CuPy's raw output format `[H, W, num_views]`, rather than the PyTorch-standard `[num_views, H, W]`. This caused the MSE loss to compare the view dimension against the spatial width dimension.
    2. **Physical Ray Tracing Logic**: During the `low_res_stage`, the script was evaluating the Gaussians on a downsampled grid (e.g. `268x268` spatially). However, because `ChestXSim` simulates physical X-rays hitting a physical hardware detector (`2144x2144` binned 4x to `536x536`), the generated projection will *always* be `536x536`, regardless of how low-res the inputted volume is. The loss function crashed trying to compare the full `536x536` projection against the low-res `268x268` ground truth.
* **The Fix**: 
    1. Added `.permute(2, 0, 1).contiguous()` to the return statement of `ct_projector.py` to fix the view dimension order.
    2. Updated the MSE loss function in `train_medlift3d.py` during the `low_res_stage` to explicitly slice the generated projection by 2 (`train_projs[:, ::2, ::2]`), bringing the physical ray-traced projection down to the `268x268` scale required to match the ground truth.

---

## 12. Stage 3: Kaggle Progress Bar (Tqdm) Display Truncation
**Files Modified**: `train_medlift3d.py`
* **Problem**: The `tqdm` progress bar appeared "stuck" and was missing the ETA and `it/s` speed statistics on the right side.
* **The Fix**: The LUNA25 `case_id` string is over 60 characters long. Kaggle's narrow console width forced `tqdm` to drop the statistics to fit the `desc` string on a single line. Truncating the description to `desc=f"Optimizing {case_id[:13]}..."` freed up terminal space, allowing `tqdm` to render the full progress statistics properly.

---

## 13. Stage 3: Kaggle 12-Hour Session Timeout and Checkpointing
**Files Modified**: `configs/medlift3d.yaml`, `train_medlift3d.py`, `visualize_diffusion.py`
* **Problem**: 15 physical X-Ray simulated forward projections takes ~2.8 seconds per iteration on a Kaggle T4 GPU. At 15,000 iterations (the default for random initialization), the script would take ~19 hours to run. Kaggle enforces a hard 12-hour timeout, meaning the kernel would die and drop 100% of the progress before the final `.npy` save block triggered.
* **The Fix**: 
    1. **Iterative Reduction**: Because the `BiFlowNet` diffusion prior provides a highly accurate structural starting point, 15,000 iterations is massive overkill. We reduced `max_iter` down to **8,000** and extended `low_reso_stage` to **6,000**. This successfully brings the total runtime down to an estimated 7.9 hours, comfortably clearing the Kaggle timeout.
    2. **Loop Checkpointing**: Injected a safety block inside the `train_medlift3d.py` loop to run `np.save` every 2,000 iterations, ensuring that intermediate reconstructions are written to the `results/` folder in case of sudden kernel termination.
    3. **Idle GPU Utilization**: Wrote a standalone `visualize_diffusion.py` script that maps to `cuda:1` (the idle second GPU on Kaggle), allowing the user to generate slices of the raw Diffusion Prior output simultaneously without causing OOM crashes on the training GPU.
