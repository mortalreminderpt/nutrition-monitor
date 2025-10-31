import os
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from rgbd_bifpn_unet import (
    parse_args, 
    prepare_dataloaders,
    build_model, 
    load_checkpoint, 
    TargetNormalizer,
    Nutrition5KMultitaskDataset,
)

# fix seed
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = Path("comp-90086-nutrition-5-k") / "Nutrition5K" / "Nutrition5K"
checkpoint_path = Path("best_model.pt")
output_csv = Path("submission.csv")

# manually generate test IDs
test_ids = [f"dish_{i}" for i in range(3301, 3491)]

# prepare dataloaders to get normalizer and other stats
config = parse_args()
_, val_loader, target_normalizer, calorie_bins, volume_lookup, volume_stats, _ = prepare_dataloaders(config)

# create test dataset
test_dataset = Nutrition5KMultitaskDataset(
    test_ids,
    data_root=data_root,
    input_size=config.input_size,
    train=False,
    target_normalizer=target_normalizer,
    calorie_bins=calorie_bins,
    volume_lookup=volume_lookup,
    volume_stats=volume_stats,
    labels={},
    macros=None,
    depth_threshold=config.depth_threshold,
    canonical_orientation=config.canonical_orientation,
)

def collate_fn(batch):
    """
    Custom collate function to handle string dish_ids.
    """
    rgb = torch.stack([item["rgb"] for item in batch])
    depth = torch.stack([item["depth"] for item in batch])
    dish_ids = [item["dish_id"] for item in batch]
    return {"rgb": rgb, "depth": depth, "dish_id": dish_ids}

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config.val_batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)

# build and load model
effective_bucket_count = max(0, len(calorie_bins) - 1)
model = build_model(config, bucket_count=effective_bucket_count).to(device)
_, extra = load_checkpoint(model, optimizer=None, scaler=None, path=checkpoint_path)
tn = TargetNormalizer(**extra["target_normalizer"])
denorm = lambda x: x * (tn.std + 1e-6) + tn.mean

# run inference
print(f"Running inference on {len(test_ids)} test samples...")
model.eval()
results = []

with torch.no_grad():
    for batch in test_loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        dish_ids = batch["dish_id"]
        
        out = model(rgb, depth)["total_calories"].detach().view(-1)
        pred = denorm(out)
        
        for dish_id, calories in zip(dish_ids, pred.cpu().numpy()):
            results.append({"ID": dish_id, "Value": float(calories)})

# save to csv
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Submission saved to {output_csv}")
print(f"Total predictions: {len(df)}")
print(f"Sample predictions:")
print(df.head(10))