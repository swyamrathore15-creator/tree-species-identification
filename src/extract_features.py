
import os
import numpy as np
import pandas as pd
import rasterio

INPUT_CSV = (
    r"D:\ML PROJECT\Tree Species Identification"
    r"\data\processed\balanced_dataset_split.csv"
)

OUTPUT_CSV = (
    r"D:\ML PROJECT\Tree Species Identification"
    r"\data\processed\image_features.csv"
)

df = pd.read_csv(INPUT_CSV)
records = []

def stats(values, prefix):
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }

print(f"Extracting features from {len(df)} images...")

for i, row in df.iterrows():
    try:
        with rasterio.open(row["image_path"]) as src:
            image = src.read().astype(np.float32)

        # TreeSatAI aerial band order: NIR, Green, Blue, Red
        nir, green, blue, red = image

        features = {
            "filename": row["filename"],
            "species": row["species"],
            "split": row["split"],
        }

        # Per-band statistics, normalized to 0–1
        for name, band in [
            ("nir", nir),
            ("green", green),
            ("blue", blue),
            ("red", red),
        ]:
            features.update(stats(band / 255.0, name))

        # NDVI
        denominator = nir + red
        ndvi = np.divide(
            nir - red,
            denominator,
            out=np.zeros_like(nir),
            where=denominator != 0,
        )

        # NDWI-style NIR/Green index; name it explicitly
        # rather than assuming it measures water accurately.
        green_denominator = nir + green
        nir_green_index = np.divide(
            nir - green,
            green_denominator,
            out=np.zeros_like(nir),
            where=green_denominator != 0,
        )

        features.update(stats(ndvi, "ndvi"))
        features.update(stats(nir_green_index, "nir_green_index"))

        # Fraction of pixels where NDVI denominator is zero
        features["zero_ndvi_denominator_fraction"] = float(
            np.mean(denominator == 0)
        )

        records.append(features)

    except Exception as error:
        print(f"\nFailed: {row['filename']} — {error}")

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(df)}")

feature_df = pd.DataFrame(records)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
feature_df.to_csv(OUTPUT_CSV, index=False)

print("\nFeature extraction complete.")
print("Successful images:", len(feature_df))
print("Feature columns:", len(feature_df.columns) - 3)
print("Saved to:", OUTPUT_CSV)
print("\nFeatures by species:")
print(feature_df["species"].value_counts())
