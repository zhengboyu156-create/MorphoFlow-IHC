"""Plot logged MorphoFlow losses for the current Flow-stage run."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
rows = []
for line in (ROOT / "flow.log").read_text(encoding="utf-8").splitlines():
    if not line.startswith("{"):
        continue
    record = json.loads(line)
    if record.get("stage") == "flow" and "step" in record and "flow_loss" in record:
        rows.append(record)

df = pd.DataFrame(rows).sort_values("step").drop_duplicates("step", keep="last")
df["generator_total"] = (
    df["flow_loss"] + 0.1 * df["nce_loss"] + 0.1 * df["adversarial_loss"]
)
df.to_csv(ROOT / "flow_losses.csv", index=False)

window = 11
series = [
    ("flow_loss", "Regional CFM loss"),
    ("nce_loss", "PatchNCE loss"),
    ("adversarial_loss", "Generator adversarial loss"),
    ("generator_total", "Weighted generator total = CFM + 0.1 NCE + 0.1 GAN"),
    ("critic_loss", "Discriminator loss (separate optimizer)"),
]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
axes = axes.ravel()
for ax, (col, label) in zip(axes, series):
    x = df["step"].to_numpy()
    y = df[col].to_numpy()
    smoothed = pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy()
    ax.plot(x, y, color="#8aa3b8", alpha=0.35, linewidth=0.7, label="logged value")
    ax.plot(x, smoothed, color="#145da0", linewidth=1.8, label=f"{window}-point moving mean")
    ax.set_title(label)
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", frameon=False, fontsize=8)
axes[-1].axis("off")
for ax in axes[:-1]:
    ax.set_xlabel("Optimizer step")
fig.suptitle(
    f"MorphoFlow fast algorithm — Flow-stage losses (through step {int(df.step.max()):,})\n"
    "PPO has not started; generator total is reconstructed from logged components",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(ROOT / "flow_losses.png", dpi=180, bbox_inches="tight")
fig.savefig(ROOT / "flow_losses.svg", bbox_inches="tight")
print(f"rows={len(df)} first={int(df.step.min())} last={int(df.step.max())}")
print(f"saved={ROOT / 'flow_losses.png'}")
