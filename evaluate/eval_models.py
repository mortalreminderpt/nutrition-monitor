import numpy as np
import pandas as pd
from pathlib import Path

model_files = {
    "RGBD": "rgbd_history.csv",
    "RGBD+BiFPN+U-Net": "rgbd_bifpn_unet_history.csv",
    "RGBD+BiFPN+U-Net+SelfSup": "rgbd_bifpn_unet_selfsup_history.csv",
    "RGBD+SelfSup": "rgbd_selfsup_history.csv",
}
alpha = 0.2
hist_dir = Path("history")
out_path = Path("results/model_summary.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

def load_history(csv_path: Path):
    """
    load csv and return epochs and rmse
    """
    df = pd.read_csv(csv_path)
    # find rmse column as y
    rmse_col = next((c for c in ("rmse", "calorie_rmse") if c in df.columns), None)
    # only keep validation data
    df = df[df["phase"] == "val"].copy()
    return df["epoch"].to_numpy(), df[rmse_col].to_numpy()

def ema_min(y: np.ndarray):
    """
    Use EMA to find the minimum value and its epoch id
    """
    a = float(alpha)
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = a * y[i] + (1 - a) * out[i - 1]
    idx = int(np.argmin(out))
    return float(out[idx]), idx

def calculate(x: np.ndarray):
    """
    Calculate mean, std and 95% confidence interval
    """
    x = np.asarray(x, float)
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    half = 1.96 * sd / np.sqrt(n)
    return mean, sd, (f"{mean-half:.2f}", f"{mean+half:.2f}")

# collect data from all seed csv files
rows = []
for seed_dir in sorted(p for p in hist_dir.iterdir() if p.is_dir() and p.name.startswith("seed")):
    seed = seed_dir.name
    for label, fname in model_files.items():
        csv_path = seed_dir / fname
        if not csv_path.exists(): 
            continue
        epochs, rmse = load_history(csv_path)
        val, idx = ema_min(rmse)
        rows.append({
            "seed": seed,
            "model": label,
            "rmse_ema": val,
            "at_epoch": int(epochs[idx]),
        })

# calculate statistics
df = pd.DataFrame(rows)
summ_rows = []
for model, g in df.groupby("model"):
    vals = g["rmse_ema"].to_numpy(float)
    mean, sd, (lo, hi) = calculate(vals)
    summ_rows.append({
        "model": model,
        "num_seeds": int(g.shape[0]),
        "rmse_mean": round(mean, 4),
        "rmse_std": round(sd, 4),
        "rmse_ci95": f"[{lo},{hi}]",# if isinstance(lo, str) else f"[{lo:.2f},{hi:.2f}]",
        "rmse_best": round(float(np.min(vals)), 4),
        "rmse_median": round(float(np.median(vals)), 4),
        "median_at_epoch": int(g["at_epoch"].median()),
    })

# save
summary = pd.DataFrame(summ_rows).sort_values("rmse_mean")
summary.to_csv(out_path, index=False)
print(summary.to_string(index=False))
print(f"\nSaved: {out_path}")
