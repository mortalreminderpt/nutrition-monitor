# Seeing Beyond the Plate: A Robust RGB‑Depth Model for Food Calorie Estimation

<img src="https://raw.githubusercontent.com/mortalreminderpt/just-a-pic/refs/heads/main/1761108209967.jpg" alt="Nutrition5K Leaderboard (20 Oct 2025)" />

## 0. Introduction

This repo contains:

- Training code for 4 models: 
    - RGBD, 
    - RGBD+SelfSup, 
    - RGBD+BiFPN+U‑Net, 
    - RGBD+BiFPN+U‑Net+SelfSup.
- Clean evaluation scripts: 
    - model comparison, 
    - training curves, 
    - noise robustness,
    - best/worst cases, 
    - point clouds, 
    - submission file maker.
- Train history of all 5 seeds.
- Reproducible seeds and folder layout.
- No external data or pre‑trained weights.


## 1. Setup

We use [uv](https://docs.astral.sh/uv/) for fast, lockfile‑based Python envs.

```bash
# 1) Install dependencies
uv sync

# 2) Activate
source .venv/bin/activate

# 3) Download dataset (Kaggle CLI)
kaggle competitions download -c comp-90086-nutrition-5-k

# 4) Unzip
unzip comp-90086-nutrition-5-k.zip -d comp-90086-nutrition-5-k/
```

### 1.1 Python versions

* Core training and evaluation: Python 3.13 (for compatibility of pytorch with cuda).
* Open3D (for point cloud) does not yet support 3.13. So we use a small extra env with Python 3.12 ONLY for the point cloud step:

```bash
# only for open3d
python3.12 -m venv .venv-open3d
source .venv-open3d/bin/activate
pip install open3d
```

Switch back to `.venv` for all normal tasks.

## 2. Training

All models are trained from scratch (no pre‑trained weights). We set seeds for reproducibility. Different identical graphics cards will still lead to slightly different results, but the results are basically the same.

### 2.1 Train RGBD baseline

```bash
python -m rgbd --self-sup-epochs 0 --output-dir artifacts/rgbd_baseline
# If you do not set output-dir, the default is artifacts/rgbd_selfsup.
```

### 2.2 Train RGBD+SelfSup

```bash
python -m rgbd --self-sup-epochs 20
# Default output-dir: artifacts/rgbd_selfsup.
```

### 2.3 Train RGBD+BiFPN+U‑Net

```bash
python -m rgbd_bifpn_unet --self-sup-epochs 0 --output-dir artifacts/bifpn_unet_test
# If you do not set output-dir, the default is artifacts/rgbd_bifpn_unet_selfsup.
```

### 2.4 Train RGBD+BiFPN+U‑Net+SelfSup

```bash
python -m rgbd_bifpn_unet --self-sup-epochs 20
# Default output-dir: artifacts/rgbd_bifpn_unet_selfsup.
```

## 3. Evaluation

Our training logs are already included in the repo for quick reproduction. If you want to evaluate your own results from training, you should copy the rsults to the history folder and rename them like our logs.

### 3.1 Compare models (for Table II)

Compute mean, std, CI, best, and median RMSE over 5-folds in multiple seeds.

```bash
python -m evaluate.eval_models
```

* Output: `model_summary.csv` and a pretty table in stdout.
* What we show: Mean RMSE ± Std, 95% CI, Best RMSE, Median RMSE, and the epoch of the median.

### 3.2 Plot training curves (for Fig. 2)

Visualize valid RMSE over epochs for all 4 models. We also draw smoothed lines.

```bash
python -m evaluate.plot_training_curves
```

* Input: histories in `history/`.
* Output: `history_rmse_curves.pdf`.

### 3.3 Random noise tests (for Table III & Fig. 3)

Add Gaussian noise to depth images to test robustness. Also can switch to color noise.

1. Evaluate with injected noise

```bash
python -m evaluate.eval_depth_noise   # depth by default
```

2. Plot the noise curves

```bash
python -m evaluate.plot_depth_noise
```

3. Switch modality (optional) Open both files and change:

```python
eval_type = "depth"  # -> "color"
```

Then run the 2 commands again.

### 3.4 Extreme image cases (for Table IV & Fig. 4a)

Find the samples with the largest absolute error or the best ones if you set `best=True`.

```bash
python -m evaluate.eval_worst_cases   # default: best=False
```

### 3.5 Point clouds (for Fig. 4a&b)

We build point clouds for the best and worst samples to inspect geometry quality.

```bash
source .venv-open3d/bin/activate
python -m evaluate.eval_pcd
```

Then, screenshot to get Fig. 4a&b.

### 3.6 Make a Kaggle submission

Run the model on test set and output `submission.csv`.

```bash
python -m evaluate.generate_submission
```

## 5. Results you should see

### History (logs)

- `history/seed*/*_history.csv`
- `history/seed*/*_history.jsonl`: logs per epoch

### Figures

- `figs/history_rmse_curves.pdf`: Fig. 2
- `figs/depth_noise_curve.pdf`: Fig. 3a
- `figs/color_noise_curve.pdf`: Fig. 3b

### Tables

- `results/model_summary.csv`: Table II
- `results/depth_noise_results.csv`: Table III
- `results/color_noise_results.csv`: Not used in report, for generate Fig. 3b

### Submission

- `submissions/submission.csv`: final Kaggle submission.
