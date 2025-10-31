import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

model_files = {
    "RGBD": "rgbd_history.csv",
    "RGBD+BiFPN+U-Net": "rgbd_bifpn_unet_history.csv",
    "RGBD+BiFPN+U-Net+SelfSup": "rgbd_bifpn_unet_selfsup_history.csv",
    "RGBD+SelfSup": "rgbd_selfsup_history.csv",
}
alpha = 0.2
hist_dir = Path("history/seed42")
out_path = Path("figs/history_rmse_curves.pdf")
out_path.parent.mkdir(parents=True, exist_ok=True)

def load_curve(csv_path: Path):
    """
    load csv and preprocess to curve data
    """
    df = pd.read_csv(csv_path)
    # find rmse column as y
    rmse_col = next((c for c in ("rmse", "calorie_rmse") if c in df.columns), None)
    # only keep validation data
    df = df[df["phase"] == "val"].copy()
    return df, rmse_col

def smooth(y: np.ndarray, alpha: float):
    """
    Exponential Moving Average (EMA) with alpha as smoothing factor
    """
    a = float(alpha)
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = a * y[i] + (1 - a) * out[i - 1]
    arr = np.asarray(out, dtype=float)
    out = np.minimum.accumulate(arr)
    return out

plt.figure(figsize=(12.8, 4.8))

# fix color like other plots
prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
cycle_colors = prop_cycle.by_key()["color"] if prop_cycle else None

# loop each model
for idx, label in enumerate(sorted(model_files)):
    csv_path = hist_dir / model_files[label]
    df, rmse_col = load_curve(csv_path)
    epochs = df["epoch"].to_numpy()
    rmse = df[rmse_col].to_numpy()
    smoothed_rmse = smooth(rmse, alpha)

    color = (cycle_colors[idx % len(cycle_colors)]) if cycle_colors else None
    # original
    plt.plot(epochs, rmse, linestyle="dotted", linewidth=1, alpha=0.5, color=color)
    # smoothed
    plt.plot(epochs, smoothed_rmse, label=label, linewidth=1.5, color=color)

plt.xlabel("Epoch")
plt.ylabel("RMSE (kcal)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
print(f"Figure saved: {out_path}")