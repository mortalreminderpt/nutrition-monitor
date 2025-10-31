from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# config
eval_type = "color"  # "depth" or "color"
input_csv = Path(f"results/{eval_type}_noise_results.csv")
output_fig = Path(f"figs/{eval_type}_noise_curve.pdf")

# load data
df = pd.read_csv(input_csv)
df["sigma"] = df["sigma"].astype(float)
df["rmse"] = df["rmse"].astype(float)

# use latest run_id
latest_run = df.groupby("run_id")["timestamp"].max().sort_values().index[-1]
data = df[df.run_id == latest_run]
print(f"Plotting run_id: {latest_run}")

# plot
output_fig.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(6.4, 4.8))
markers = ["o", "D", "s", "^", "v", "P", "X", "*"]
linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

for i, (label, group) in enumerate(sorted(data.groupby("label"), key=lambda x: x[0])):
    group = group.sort_values("sigma")
    plt.plot(group.sigma.values, group.rmse.values,
             marker=markers[i % len(markers)],
             linestyle=linestyles[i % len(linestyles)],
             label=label)

x_label = "Depth" if eval_type == "depth" else "Color"
plt.xlabel(f"{x_label} noise std $\sigma$")
plt.ylabel("RMSE (kcal)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(output_fig, bbox_inches="tight", pad_inches=0.02)
print(f"Figure saved: {output_fig}")