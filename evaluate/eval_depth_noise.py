import os
import random
import math
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone
from rgbd_bifpn_unet import (
    parse_args, 
    prepare_dataloaders, 
    build_model, 
    load_checkpoint, 
    TargetNormalizer,
)
from rgbd import (
    parse_args as parse_args_baseline,
    prepare_dataloaders as prepare_dataloaders_baseline,
    build_model as build_model_baseline,
    load_checkpoint as load_checkpoint_baseline,
)

# fix seed
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# config
eval_type = "depth"  # "depth" or "color"
output_csv = Path(f"results/{eval_type}_noise_results.csv")
sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%input_size")
timestamp = datetime.now(timezone.utc).isoformat()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = Path("comp-90086-nutrition-5-k") / "Nutrition5K" / "Nutrition5K"

# prepare model and dataloader

# baseline
config_baseline = parse_args_baseline()
model_baseline, input_size = build_model_baseline()
model_baseline = model_baseline.to(device)
_, val_baseline, _ = prepare_dataloaders_baseline(config_baseline, input_size)

# bifpn
config = parse_args()
_, val_loader, target_normalizer, calorie_bins, *_ = prepare_dataloaders(config)
bifpn_model = build_model(config, bucket_count=max(0, len(calorie_bins)-1)).to(device)

# evaluation functions
@torch.no_grad()
def eval_baseline(model, dl, sigma):
    model.eval()
    s, n = 0.0, 0
    for x, y in dl:
        x = x.to(device, non_blocking=True).clone()
        y = y.to(device, non_blocking=True).view(-1)
        if eval_type == "depth" and sigma > 0:
            x[:, 3:4] += torch.randn_like(x[:, 3:4]) * sigma
        elif eval_type == "color" and sigma > 0:
            x[:, 0:3] += torch.randn_like(x[:, 0:3]) * sigma
        p = model(x)
        p = getattr(p, "logits", p)
        if isinstance(p, (tuple, list)):
            p = p[0]
        p = p.squeeze(-1)
        if p.dim() > 1:
            p = p[:, 0]
        p = p.view(-1)
        d = p - y
        s += torch.sum(d * d).item()
        n += d.numel()
    return math.sqrt(s / n) if n > 0 else float("nan")

@torch.no_grad()
def eval_bifpn(model, dl, denorm, sigma):
    model.eval()
    s, n = 0.0, 0
    for batch in dl:
        rgb = batch["rgb"].to(device, non_blocking=True).clone()
        depth = batch["depth"].to(device, non_blocking=True).clone()
        if eval_type == "depth" and sigma > 0:
            depth[:, 3:4] += torch.randn_like(depth[:, 3:4]) * sigma
        elif eval_type == "color" and sigma > 0:
            rgb[:, 0:3] += torch.randn_like(rgb[:, 0:3]) * sigma
        targets = batch.get("targets", None)
        if targets is None:
            continue
        out = model(rgb, depth)["total_calories"].detach().view(-1)
        pred = denorm(out)
        gt = targets["total_calories"].to(device, non_blocking=True).view(-1)
        diff = pred - gt
        s += torch.sum(diff * diff).item()
        n += diff.numel()
    return math.sqrt(s / n) if n > 0 else float("nan")

# write results
output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open("w", encoding="utf-8") as f:
    f.write("run_id,timestamp,label,sigma,rmse\n")
    
    # 1. RGBD baseline
    label = "RGBD"
    ckpt = Path("rgbd_best.pt")
    if ckpt.exists():
        load_checkpoint_baseline(ckpt, model_baseline, optimizer=None, scaler=None)
        for sigma in sigmas:
            rmse = eval_baseline(model_baseline, val_baseline, sigma)
            f.write(f"{run_id},{timestamp},{label},{sigma:.6f},{rmse:.6f}\n")
            print(f"[{label} | Sigma={sigma:.3f}] RMSE={rmse:.4f} kcal")
    
    # 2. RGBD+SelfSup baseline
    label = "RGBD+SelfSup"
    ckpt = Path("rgbd_selfsup_best.pt")
    if ckpt.exists():
        load_checkpoint_baseline(ckpt, model_baseline, optimizer=None, scaler=None)
        for sigma in sigmas:
            rmse = eval_baseline(model_baseline, val_baseline, sigma)
            f.write(f"{run_id},{timestamp},{label},{sigma:.6f},{rmse:.6f}\n")
            print(f"[{label} | Sigma={sigma:.3f}] RMSE={rmse:.4f} kcal")
    
    # 3. RGBD+BiFPN+U-Net
    label = "RGBD+BiFPN+U-Net"
    ckpt = Path("rgbd_bifpn_unet_best.pt")
    if ckpt.exists():
        _, extra = load_checkpoint(bifpn_model, optimizer=None, scaler=None, path=ckpt)
        tn = TargetNormalizer(**extra["target_normalizer"])
        denorm = lambda x: x * (tn.std + 1e-6) + tn.mean
        for sigma in sigmas:
            rmse = eval_bifpn(bifpn_model, val_loader, denorm, sigma)
            f.write(f"{run_id},{timestamp},{label},{sigma:.6f},{rmse:.6f}\n")
            print(f"[{label} | Sigma={sigma:.3f}] RMSE={rmse:.4f} kcal")
    
    # 4. RGBD+BiFPN+U-Net+SelfSup
    label = "RGBD+BiFPN+U-Net+SelfSup"
    ckpt = Path("rgbd_bifpn_unet_selfsup_best.pt")
    if ckpt.exists():
        _, extra = load_checkpoint(bifpn_model, optimizer=None, scaler=None, path=ckpt)
        tn = TargetNormalizer(**extra["target_normalizer"])
        denorm = lambda x: x * (tn.std + 1e-6) + tn.mean
        for sigma in sigmas:
            rmse = eval_bifpn(bifpn_model, val_loader, denorm, sigma)
            f.write(f"{run_id},{timestamp},{label},{sigma:.6f},{rmse:.6f}\n")
            print(f"[{label} | Sigma={sigma:.3f}] RMSE={rmse:.4f} kcal")

print(f"CSV saved: {output_csv} (run_id={run_id})")
