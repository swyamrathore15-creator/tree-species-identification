import pandas as pd

INPUT_FILE = r"D:\ML PROJECT\Tree Species Identification\data\processed\dataset_index.csv"

OUTPUT_FILE = r"D:\ML PROJECT\Tree Species Identification\data\processed\balanced_dataset.csv"


# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original dataset:")
print(df["species"].value_counts())


# Find smallest class
samples_per_class = df["species"].value_counts().min()

print(f"\nSamples per class: {samples_per_class}")


# Create balanced dataset
balanced_parts = []

for species in df["species"].unique():

    species_df = df[df["species"] == species]

    sampled = species_df.sample(
        n=samples_per_class,
        random_state=42
    )

    balanced_parts.append(sampled)


# Combine all classes
balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)


# Shuffle
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Save
balanced_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# Results
print("\n" + "=" * 60)
print("BALANCED DATASET")
print("=" * 60)

print("\nSamples per species:")
print(balanced_df["species"].value_counts())

print(f"\nTotal samples: {len(balanced_df)}")

print("\nSaved to:")
print(OUTPUT_FILE)