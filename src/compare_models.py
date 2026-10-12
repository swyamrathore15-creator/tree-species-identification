
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Evaluation results from the existing runs
models = ["Random Forest", "CNN V2 Fixed"]
accuracy = [82.79, 83.77]

# RF macro F1 is estimated from its rounded per-class scores.
# CNN macro F1 is from its classification report.
macro_f1 = [82.58, 83.49]

# 1. Model performance comparison
x = np.arange(len(models))
width = 0.34

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy")
bars2 = ax.bar(x + width / 2, macro_f1, width, label="Macro F1-score")

ax.set_ylabel("Score (%)")
ax.set_title("Random Forest vs CNN V2 Fixed")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.25)
ax.legend()

ax.bar_label(bars1, fmt="%.2f%%", padding=3)
ax.bar_label(bars2, fmt="%.2f%%", padding=3)

fig.tight_layout()
fig.savefig(RESULTS / "model_comparison.png", dpi=180)
plt.close(fig)

# 2. Confusion matrices
classes = ["Fagus", "Picea", "Pinus", "Quercus"]

rf_cm = np.array([
    [63, 3, 5, 11],
    [4, 68, 4, 1],
    [1, 3, 71, 6],
    [9, 5, 1, 53],
])

cnn_cm = np.array([
    [67, 3, 1, 11],
    [3, 64, 9, 1],
    [0, 5, 75, 1],
    [9, 1, 6, 52],
])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, cm, title in zip(
    axes,
    [rf_cm, cnn_cm],
    ["Random Forest (82.79%)", "CNN V2 Fixed (83.77%)"],
):
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticklabels(classes)

    for i in range(4):
        for j in range(4):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Confusion Matrices - Official Test Set")
fig.tight_layout()
fig.savefig(
    RESULTS / "confusion_matrix_comparison.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(fig)

# 3. Summary table
summary = pd.DataFrame({
    "Model": models,
    "Test samples": [308, 308],
    "Accuracy (%)": accuracy,
    "Macro F1-score (%)": macro_f1,
    "Macro F1 note": [
        "Calculated from test predictions",
        "From classification report",
    ],
})

summary.to_csv(RESULTS / "model_comparison.csv", index=False)

print("Comparison files saved:")
print("results/model_comparison.png")
print("results/confusion_matrix_comparison.png")
print("results/model_comparison.csv")
print()
print(summary.to_string(index=False))
