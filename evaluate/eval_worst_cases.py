import os
import random
import numpy as np
import torch
import pandas as pd
from rgbd_bifpn_unet import parse_args, prepare_dataloaders, build_model, TargetNormalizer, load_checkpoint
from pathlib import Path

best = True # False to select worst cases

# fix seed
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

checkpoint = Path("rgbd_bifpn_unet_best.pt")
output_dir = Path(f"results/{'best' if best else 'worst'}_images.csv")
config = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare dataset and model (seed have been fixed to 42 in config)
_, val_loader, _, target_normalizer, calorie_bins, *_ = prepare_dataloaders(config)
model = build_model(config, bucket_count=max(0, len(calorie_bins)-1)).to(device).eval()
_, extra = load_checkpoint(model, optimizer=None, scaler=None, path=checkpoint)
tn = TargetNormalizer(**extra["target_normalizer"])
denorm = lambda x: x*(tn.std+1e-6)+tn.mean

# collect err of each image
rows = []
with torch.no_grad():
    for batch in val_loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        out = model(rgb, depth)["total_calories"].detach().cpu().view(-1)
        pred = denorm(out)
        gt = batch["targets"]["total_calories"].view(-1)
        err = (pred - gt).abs().numpy()  # err = abs(pred - gt) = sqrt(MSE)
        ids = batch["dish_id"]
        rows.extend([{"ID": i, "AE": float(e)} for i, e in zip(ids, err)])

# save top k cases to csv
pd.DataFrame(rows).sort_values("AE", ascending=best).head(10).to_csv(output_dir, index=False)
print(f"Saved top-k cases to {output_dir}")