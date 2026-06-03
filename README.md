# Conditional Diffusion

A from-scratch PyTorch implementation of a **class-conditional Denoising Diffusion Probabilistic Model (DDPM)** for MNIST, built around a custom U-Net and supporting:

- **Classifier-Free Guidance (CFG)** with a learnable null class.
- **DDIM** accelerated sampling (50 steps) in addition to full DDPM (1000 steps).
- **Exponential Moving Average (EMA)** of model weights for higher-quality samples.
- **Cosine noise schedule**, mixed-precision (bfloat16) training, gradient accumulation, gradient clipping, and OneCycle learning-rate scheduling.
- An interactive **Gradio** demo to watch the denoising process in real time.
- **FID** and **Inception Score** evaluation across all training checkpoints.
- A bonus **ControlNet + Stable Diffusion v1.5** pipeline that uses generated MNIST digits as scribble guidance to produce stylized, text-prompted variants.

The codebase is small (~Python only, no training scaffolding library) and is designed to run on **Apple Silicon (MPS)** with a CPU fallback.

---

## Table of contents

1. [Repository structure](#repository-structure)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Inference and generation](#inference-and-generation)
   - [CLI: `test.py`](#cli-testpy)
   - [Gradio app: `app.py`](#gradio-app-apppy)
   - [Stable Diffusion stylization: `test_sd.py`](#stable-diffusion-stylization-test_sdpy)
6. [Evaluation](#evaluation)
7. [Implementation details](#implementation-details)
8. [Hardware notes](#hardware-notes)

---

## Repository structure

```
conditional-diffusion/
├── diffusion_core/                  # Core library (model + diffusion utilities)
│   ├── model.py                     # DiffusionModel (U-Net backbone)
│   ├── schedule.py                  # get_cosine_schedule(T)
│   ├── sampling.py                  # ddpm_sample(...) and ddim_sample(...)
│   ├── ema.py                       # EMA wrapper for model weights
│   └── utils.py                     # save_training_checkpoint(...)
├── config.yaml                      # Centralized hyperparameters
├── train.py                         # Training loop (TensorBoard, EMA, CFG, AMP)
├── test.py                          # CLI: generate a digit + denoising GIF
├── app.py                           # Gradio web UI for interactive generation
├── test_sd.py                       # ControlNet + SD1.5 stylization pipeline
├── evaluate.py                      # FID / IS evaluation across checkpoints
└── .gitignore
```

---

## Installation

The project targets Python 3.10+ and PyTorch with MPS support (Apple Silicon). It will fall back to CPU automatically if MPS is unavailable.

```bash
git clone https://github.com/andreaunitn/conditional-diffusion.git
cd conditional-diffusion

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install torch torchvision
pip install pyyaml tensorboard imageio matplotlib Pillow tqdm

# For the Gradio interactive demo
pip install gradio

# For evaluation (FID / Inception Score)
pip install torchmetrics torch-fidelity

# For the Stable Diffusion stylization script
pip install diffusers transformers accelerate safetensors
```

> **Note on devices.** All scripts auto-select `mps` if `torch.backends.mps.is_available()`; otherwise they fall back to `cpu`. If you adapt the code for CUDA, swap `"mps"` → `"cuda"` in the device line at the top of each script.

The MNIST dataset is downloaded automatically by `torchvision.datasets.MNIST` into `./data/` on first run.

---

## Configuration

All hyperparameters live in [`config.yaml`](./config.yaml):

```yaml
experiment_name: "dm_mnist_v4"

model:
  image_size: 32           # MNIST is resized 28 -> 32
  in_channels: 1           # grayscale
  model_channels: 96       # base channel width of the U-Net
  bottleneck_dim: 4        # spatial dim at the U-Net bottleneck
  max_channels: 512        # channel cap at the deepest level
  out_dim: 1               # predicted-noise channels
  time_emb_dim: 256        # sinusoidal time-embedding dimension
  num_classes: 10          # digits 0-9 (index 10 is reserved for the null class)

train:
  batch_size: 128
  grad_acc_steps: 4        # effective batch size: 128 * 4 = 512
  epochs: 50
  lr: 0.0002
  timesteps: 1000          # T for the forward process
  weight_decay: 0.0001
  save_dir: "checkpoints"

sample:
  T: 1000
  ddim_steps: 50           # DDIM accelerated steps
  guidance_scale: 1.2      # CFG weight
  null_class_idx: 10       # placeholder class injected for unconditional path

schedule:
  type: "cosine"
```

The `experiment_name` is used as both the TensorBoard run name (`runs/<experiment_name>`) and as the checkpoint filename prefix (`checkpoints/<experiment_name>_<epoch>.pth`).

---

## Training

Launch training with the default config:

```bash
python train.py --config config.yaml
```

What happens under the hood:

- **Data.** MNIST is loaded, resized to 32×32, and normalized to `[-1, 1]`.
- **Forward diffusion.** For each batch, timesteps are sampled uniformly and noise is added via
  `x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε`.
- **Classifier-Free Guidance.** With probability `0.1` per sample, the class label is replaced with the null-class index (`num_classes = 10`), training a single model to act as both conditional and unconditional denoiser.
- **Objective.** Standard ε-prediction MSE loss between predicted and true noise.
- **Optimization.** AdamW with weight decay 1e-4, OneCycleLR schedule (max LR `2e-4`), 4-step gradient accumulation, gradient clipping at norm 1.0, mixed precision via `torch.amp.autocast` (bfloat16).
- **EMA.** A shadow copy of the model is maintained and used for validation sampling.
- **Logging.** Training loss is written to TensorBoard every 50 steps; validation samples (one per class, generated with DDIM) are logged every 5 epochs.
- **Checkpointing.** A full checkpoint (`model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `ema_model_state_dict`, `ema_step`, `epoch`, `global_step`, `config`) is saved every 5 epochs to `checkpoints/<experiment_name>_<epoch>.pth`, plus a final `checkpoints/diffusion_model_final.pth`.

### Resuming training

```bash
python train.py --config config.yaml --resume checkpoints/dm_mnist_v4_24.pth
```

The script restores the optimizer, the AMP gradient scaler, the EMA state, the epoch counter, and the global step.

### Monitoring

```bash
tensorboard --logdir runs/
```

Open the TensorBoard URL in your browser to inspect `Loss/train` and the periodic 5×2 grid of validation samples (`Validation/Generates_Images`).

---

## Inference and generation

### CLI: `test.py`

Generate a single digit and save both the final image and a GIF of the full denoising trajectory:

```bash
python test.py \
  --model_path checkpoints/diffusion_model_final.pth \
  --digit 7 \
  --sampling ddim \
  --seed 42
```

| Flag           | Type | Default                                       | Description                                                                 |
| -------------- | ---- | --------------------------------------------- | --------------------------------------------------------------------------- |
| `--model_path` | str  | `checkpoints/diffusion_model_final.pth`       | Path to the trained checkpoint (EMA weights are loaded if available).       |
| `--digit`      | int  | random                                        | The MNIST class (0-9) to generate. If omitted, a random class is chosen.    |
| `--sampling`   | str  | `ddim`                                        | Either `ddim` (fast, 50 steps) or `ddpm` (slow, 1000 steps).                |
| `--seed`       | int  | None                                          | Random seed for reproducible generation.                                    |

**Outputs**

- `result.png` — final 32×32 grayscale digit.
- `result.gif` — animated denoising trajectory (every step for DDIM, every 20th step for DDPM).

### Gradio app: `app.py`

Launch the interactive web UI:

```bash
python app.py --model_path checkpoints/diffusion_model_final.pth
```

The interface exposes:

- **Digit** slider (0–9).
- **Seed** field (leave blank for random).
- **CFG** slider (0–9) for guidance strength.
- **Use fast sampling (DDIM)** checkbox to switch between DDIM (50 steps) and DDPM (1000 steps).
- A live **gallery** that streams intermediate denoising frames.
- A **Final Result** panel showing the predicted clean image (`x̂_0`) at the last step, upsampled to 256×256 with nearest-neighbor interpolation.

### Stable Diffusion stylization: `test_sd.py`

This script chains the small MNIST DDPM with a much larger pretrained pipeline: it generates a digit, then uses that digit as a **ControlNet scribble** condition to drive **Stable Diffusion v1.5** with a text prompt.

```bash
python test_sd.py \
  --model_path checkpoints/diffusion_model_final.pth \
  --digit 3 \
  --prompt "a glowing neon number floating in space, cyberpunk style" \
  --seed 0
```

Models downloaded from the Hugging Face Hub:

- `lllyasviel/sd-controlnet-scribble`
- `runwayml/stable-diffusion-v1-5`

The pipeline uses the `UniPCMultistepScheduler`, 25 inference steps, guidance scale 7.5, and `controlnet_conditioning_scale=1.0`. Both fp16 weights are loaded; attention slicing is enabled to keep memory usage modest.

**Outputs**

- `base_digit.png` — the raw MNIST-style digit produced by your trained model.
- `stylized_digit.png` — the SD/ControlNet-stylized 512×512 image guided by your prompt.

---

## Evaluation

Compute **FID** (Fréchet Inception Distance) and **Inception Score (IS)** for every checkpoint in a directory:

```bash
python evaluate.py \
  --checkpoint_dir checkpoints \
  --n_samples 10000
```

| Flag                | Type | Default       | Description                                                          |
| ------------------- | ---- | ------------- | -------------------------------------------------------------------- |
| `--checkpoint_dir`  | str  | `checkpoints` | Directory containing one or more `*.pth` files.                      |
| `--n_samples`       | int  | `10000`       | Number of generated samples per checkpoint used for FID/IS.          |

The script:

1. Pre-computes real-image FID statistics on the MNIST test set (replicated to 3 channels so it can flow through the InceptionV3 feature extractor at 2048-dim features).
2. For each checkpoint (sorted by trailing number in the filename), generates `n_samples` images in batches of 50 using DDIM with the saved CFG settings.
3. Loads EMA weights when available, otherwise the standard weights.
4. Prints a per-checkpoint table and the best checkpoint by FID:

```
================================================================================
Checkpoint                          | FID (Lower=Better)   | IS (Higher=Better)
--------------------------------------------------------------------------------
dm_mnist_v4_4.pth                   | ...                  | ...
dm_mnist_v4_9.pth                   | ...                  | ...
...
--------------------------------------------------------------------------------
Best model by FID: dm_mnist_v4_XX.pth (X.XXXX)
```

FID/IS are computed on CPU (`torchmetrics` backend) for stability; memory is freed between checkpoints via `gc.collect()`.

---

## Implementation details

### U-Net backbone (`diffusion_core/model.py`)

A standard residual U-Net that takes `(x_t, t, y)` and predicts the noise ε. Channel widths are controlled by `model_channels`, `max_channels`, and the bottleneck spatial dimension `bottleneck_dim`. Timesteps are encoded with sinusoidal embeddings projected to `time_emb_dim`. Class labels are embedded with `num_classes + 1` entries — the extra slot is the **null class** used during CFG dropout.

### Noise schedule (`diffusion_core/schedule.py`)

A cosine `β_t` schedule following Nichol & Dhariwal (2021), which keeps `ᾱ_t` from collapsing too quickly near the end of the chain and tends to produce better samples than a linear schedule on small images.

### Sampling (`diffusion_core/sampling.py`)

Both samplers are implemented as Python generators that **yield `(x_t, x̂_0)` at every step**, which is what makes the live denoising gallery in the Gradio app and the GIF in `test.py` possible.

- **`ddpm_sample(...)`** — ancestral sampling for the full reverse chain (1000 steps).
- **`ddim_sample(...)`** — deterministic implicit sampler, by default 50 steps. Supports an optional `seed` for reproducibility and the CFG-style guidance:

  ```
  ε̂ = ε_uncond + s · (ε_cond − ε_uncond)
  ```

  where `s = guidance_scale` and the unconditional branch is obtained by feeding the null class index.

### EMA (`diffusion_core/ema.py`)

A lightweight exponential moving average maintained on a shadow copy of the model after every optimizer step. The EMA weights are saved alongside the regular weights and are preferred for evaluation and inference.

### Checkpoint format

Each checkpoint is a Python dict containing:

```python
{
  "epoch":               int,
  "global_step":         int,
  "model_state_dict":    {...},
  "optimizer_state_dict":{...},
  "scaler_state_dict":   {...},   # AMP grad scaler
  "ema_model_state_dict":{...},
  "ema_step":            int,
  "config":              {...},   # the full config.yaml at training time
}
```

This makes every checkpoint self-describing: `app.py`, `test.py`, `evaluate.py`, and `test_sd.py` all rebuild the model architecture from `checkpoint["config"]["model"]` rather than relying on external state.

---

## Hardware notes

- **Apple Silicon (MPS).** This is the primary target. `torch.amp.autocast(device_type="mps", dtype=torch.bfloat16)` is used during training; the `GradScaler` is created on the active device.
- **CPU fallback.** All scripts will run on CPU if MPS is unavailable, but training will be slow.
- **CUDA.** The code is not CUDA-specific. To run on an NVIDIA GPU, change the device-selection line at the top of each script (`device = "mps" if ... else "cpu"`) to also check `torch.cuda.is_available()`, and adjust `autocast(device_type=...)` accordingly.
- **Memory.** With `batch_size=128` and `grad_acc_steps=4`, peak memory is modest enough to fit on most M-series Macs. For SD1.5 stylization (`test_sd.py`), fp16 weights and attention slicing are enabled by default.

---

*Maintainer: [@andreaunitn](https://github.com/andreaunitn)*
