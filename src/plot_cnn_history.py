
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = ROOT / "results" / "cnn_v2_fixed_history.csv"
RESULTS_DIR = ROOT / "results"

# Load training history
if not HISTORY_FILE.exists():
    raise FileNotFoundError(f"History file not found: {HISTORY_FILE}")

history = pd.read_csv(HISTORY_FILE)

epochs = range(1, len(history) + 1)

# 1. Accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["accuracy"], marker="o",
         label="Training accuracy")
plt.plot(epochs, history["val_accuracy"], marker="o",
         label="Validation accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN V2 - Training and Validation Accuracy")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "cnn_v2_fixed_accuracy_curve.png",
    dpi=160
)
plt.close()

# 2. Loss graph
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["loss"], marker="o",
         label="Training loss")
plt.plot(epochs, history["val_loss"], marker="o",
         label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN V2 - Training and Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "cnn_v2_fixed_loss_curve.png",
    dpi=160
)
plt.close()

# Report the best validation epoch
best_epoch = history["val_loss"].idxmin()

print("Graphs saved successfully!")
print("Accuracy: results/cnn_v2_fixed_accuracy_curve.png")
print("Loss: results/cnn_v2_fixed_loss_curve.png")
print(f"Best epoch: {best_epoch + 1}")
print(f"Lowest validation loss: {history.loc[best_epoch, 'val_loss']:.4f}")
print(f"Validation accuracy at best epoch: "
      f"{history.loc[best_epoch, 'val_accuracy']:.4f}")
