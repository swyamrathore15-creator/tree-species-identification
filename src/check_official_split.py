import os
import pandas as pd

BASE = r"D:\ML PROJECT\Tree Species Identification\data\raw\TreeSatAI"

DATASET_FILE = r"D:\ML PROJECT\Tree Species Identification\data\processed\balanced_dataset.csv"

TRAIN_FILE = os.path.join(BASE, "train_filenames.lst")
TEST_FILE = os.path.join(BASE, "test_filenames.lst")


# --------------------------------------------------
# LOAD OUR BALANCED DATASET
# --------------------------------------------------

df = pd.read_csv(DATASET_FILE)

print("=" * 60)
print("OFFICIAL TRAIN / TEST SPLIT CHECK")
print("=" * 60)

print(f"\nTotal selected images: {len(df)}")


# --------------------------------------------------
# LOAD OFFICIAL FILE LISTS
# --------------------------------------------------

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_files = set(
        line.strip()
        for line in f
        if line.strip()
    )

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_files = set(
        line.strip()
        for line in f
        if line.strip()
    )


print(f"Official train filenames: {len(train_files)}")
print(f"Official test filenames:  {len(test_files)}")


# --------------------------------------------------
# CHECK WHICH OF OUR IMAGES ARE IN EACH SPLIT
# --------------------------------------------------

df["split"] = df["filename"].apply(
    lambda x:
        "train" if x in train_files
        else "test" if x in test_files
        else "unknown"
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nOur dataset by split:")
print(df["split"].value_counts())


print("\nSpecies × split:")
print(
    pd.crosstab(
        df["species"],
        df["split"]
    )
)


# --------------------------------------------------
# CHECK FOR UNKNOWN FILES
# --------------------------------------------------

unknown = df[df["split"] == "unknown"]

print(f"\nImages not found in official lists: {len(unknown)}")

if len(unknown) > 0:
    print("\nFirst unknown files:")
    print(unknown["filename"].head(10).to_string(index=False))


# --------------------------------------------------
# CHECK FOR OVERLAP
# --------------------------------------------------

overlap = train_files.intersection(test_files)

print(f"\nTrain/test filename overlap: {len(overlap)}")


# --------------------------------------------------
# SAVE UPDATED DATASET
# --------------------------------------------------

OUTPUT_FILE = r"D:\ML PROJECT\Tree Species Identification\data\processed\balanced_dataset_split.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved dataset with official split:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)