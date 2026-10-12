import os
import json
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE = r"D:\ML PROJECT\Tree Species Identification\data\raw\TreeSatAI"

LABEL_FILE = os.path.join(
    BASE,
    "labels",
    "TreeSatBA_v9_60m_multi_labels.json"
)

OUTPUT_FILE = r"D:\ML PROJECT\Tree Species Identification\data\processed\dataset_index.csv"


# --------------------------------------------------
# SPECIES WE ARE USING
# --------------------------------------------------

SPECIES = {
    "Fagus": "aerial_60m_fagus_sylvatica",
    "Picea": "aerial_60m_picea_abies",
    "Pinus": "aerial_60m_pinus_sylvestris",
    "Quercus": "aerial_60m_quercus_petraea"
}


# --------------------------------------------------
# LOAD LABELS
# --------------------------------------------------

print("Loading labels...")

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    labels = json.load(f)

print(f"Total labelled images in JSON: {len(labels)}")


# --------------------------------------------------
# CREATE INDEX
# --------------------------------------------------

records = []

for filename, label_data in labels.items():

    # We only want pure samples:
    # e.g. [["Fagus", 1.0]]

    if len(label_data) != 1:
        continue

    species, proportion = label_data[0]

    if species not in SPECIES:
        continue

    if proportion != 1.0:
        continue

    folder = SPECIES[species]

    image_path = os.path.join(
        BASE,
        folder,
        filename
    )

    # Make sure the actual TIFF exists
    if not os.path.exists(image_path):
        continue

    records.append({
        "filename": filename,
        "species": species,
        "proportion": proportion,
        "image_path": image_path
    })


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(records)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INDEX CREATED")
print("=" * 60)

print(f"\nTotal usable images: {len(df)}")

print("\nImages per species:")
print(df["species"].value_counts())

print("\nCSV saved to:")
print(OUTPUT_FILE)

print("\nFirst 5 entries:")
print(df.head())