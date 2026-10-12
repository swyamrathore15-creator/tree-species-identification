
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

CSV_PATH = (
    r"D:\ML PROJECT\Tree Species Identification"
    r"\data\processed\balanced_dataset_split.csv"
)

df = pd.read_csv(CSV_PATH)

# Pick one image from the training set
sample = df[df["split"] == "train"].iloc[0]
image_path = sample["image_path"]

print("Sample species:", sample["species"])
print("Image:", image_path)

with rasterio.open(image_path) as src:
    image = src.read()  # Shape: (bands, height, width)

    print("\nImage information")
    print("-" * 40)
    print("Shape:", image.shape)
    print("Band count:", src.count)
    print("Data types:", src.dtypes)
    print("CRS:", src.crs)

    for i in range(src.count):
        band = image[i]
        print(
            f"Band {i + 1}: "
            f"min={band.min()}, "
            f"max={band.max()}, "
            f"mean={band.mean():.2f}, "
            f"std={band.std():.2f}"
        )


# TreeSatAI aerial band order: NIR, Green, Blue, Red
nir, green, blue, red = image.astype(np.float32)

# NDVI = (NIR - Red) / (NIR + Red)
denominator = nir + red

ndvi = np.divide(
    nir - red,
    denominator,
    out=np.zeros_like(nir),
    where=denominator != 0
)

print("\nNDVI statistics")
print("Minimum:", float(ndvi.min()))
print("Maximum:", float(ndvi.max()))
print("Mean:", float(ndvi.mean()))

# Correct visible RGB order
rgb = np.stack([red, green, blue], axis=-1) / 255.0

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(np.clip(rgb, 0, 1))
axes[0].set_title(f"Correct RGB view — {sample['species']}")
axes[0].axis("off")

im = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
axes[1].set_title("NDVI")
axes[1].axis("off")

fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
