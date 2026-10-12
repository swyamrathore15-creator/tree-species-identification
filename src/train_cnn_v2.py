"""
CNN V2 for Tree Species Identification using TreeSatAI
Designed for Windows, TensorFlow 2.21, CPU training, and limited RAM.

Run from the project root:
    python src/train_cnn_v2.py

Expected project paths:
    data/processed/balanced_dataset_split.csv
    data/raw/TreeSatAI/aerial_60m_fagus_sylvatica/
    data/raw/TreeSatAI/aerial_60m_picea_abies/
    data/raw/TreeSatAI/aerial_60m_pinus_sylvestris/
    data/raw/TreeSatAI/aerial_60m_quercus_petraea/

The script keeps the official test set separate and writes V2 results to new files.
"""

from pathlib import Path
import json
import math
import os
import random
import sys

# Reduce noisy TensorFlow logs; this does not hide Python exceptions.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# -------------------- Configuration --------------------

SEED = 42
BATCH_SIZE = 8
MAX_EPOCHS = 25
PATIENCE = 5
LEARNING_RATE = 0.001
SHUFFLE_BUFFER = 256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "balanced_dataset_split.csv"
TREE_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "TreeSatAI"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = ["Fagus", "Picea", "Pinus", "Quercus"]

SPECIES_FOLDER = {
    "Fagus": "aerial_60m_fagus_sylvatica",
    "Picea": "aerial_60m_picea_abies",
    "Pinus": "aerial_60m_pinus_sylvestris",
    "Quercus": "aerial_60m_quercus_petraea",
}

# TreeSatAI aerial band order: NIR, Green, Blue, Red.
EXPECTED_IMAGE_SHAPE = (304, 304, 4)

# -------------------- Reproducibility and directories --------------------

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep CPU training stable on an 8 GB RAM laptop.
try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, (os.cpu_count() or 2) // 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass  # TensorFlow may already have initialized its runtime.


def find_column(df, candidates, required=True):
    """Return the first matching column name, case-insensitively."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise ValueError(
            f"Could not find any of these columns in {DATASET_CSV.name}: "
            f"{candidates}\nAvailable columns: {list(df.columns)}"
        )
    return None


def build_image_lookup():
    """Index TIFF paths by basename to support CSVs with filename-only entries."""
    lookup = {}
    for species in CLASS_NAMES:
        folder = TREE_DATA_DIR / SPECIES_FOLDER[species]
        if not folder.exists():
            raise FileNotFoundError(
                f"Missing image folder:\n{folder}\n"
                "Check that TreeSatAI is extracted under data/raw/TreeSatAI/."
            )
        for path in folder.rglob("*.tif"):
            lookup[path.name.lower()] = path
        for path in folder.rglob("*.tiff"):
            lookup[path.name.lower()] = path
    return lookup


def resolve_image_path(row, path_column, filename_column, species):
    """Use a CSV path if it exists; otherwise resolve the filename in its species folder."""
    if path_column is not None and pd.notna(row[path_column]):
        candidate = Path(str(row[path_column]))
        candidates = [candidate]
        if not candidate.is_absolute():
            candidates.append(PROJECT_ROOT / candidate)
            candidates.append(TREE_DATA_DIR / candidate)
        for item in candidates:
            if item.exists():
                return item

    filename = str(row[filename_column]).strip()
    direct = Path(filename)
    if direct.exists():
        return direct
    if not direct.is_absolute():
        for candidate in (PROJECT_ROOT / direct, TREE_DATA_DIR / direct):
            if candidate.exists():
                return candidate

    # Fallback is filled using the basename lookup created in main().
    return IMAGE_LOOKUP.get(Path(filename).name.lower())


def load_tiff(path):
    """Load four bands as H x W x C float32 values in [0, 1]."""
    with rasterio.open(path) as src:
        image = src.read()
    if image.ndim != 3 or image.shape[0] != 4:
        raise ValueError(f"Expected 4-band TIFF, got shape {image.shape}: {path}")
    image = np.transpose(image, (1, 2, 0)).astype(np.float32)
    if image.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(f"Expected {EXPECTED_IMAGE_SHAPE}, got {image.shape}: {path}")
    image /= 255.0
    return image


def make_dataset(frame, training=False):
    """Build a memory-efficient tf.data pipeline that loads TIFFs on demand."""
    paths = frame["resolved_path"].astype(str).to_numpy()
    labels = frame["label_id"].astype(np.int32).to_numpy()

    def generator():
        for path, label in zip(paths, labels):
            yield load_tiff(path), np.int32(label)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=EXPECTED_IMAGE_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if training:
        # Shuffle here, not in model.fit(). A modest buffer saves RAM.
        dataset = dataset.shuffle(
            buffer_size=min(SHUFFLE_BUFFER, len(frame)),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(1)
    return dataset


def build_model():
    """Small four-band CNN with light augmentation for CPU training."""
    inputs = tf.keras.Input(shape=EXPECTED_IMAGE_SHAPE, name="four_band_image")

    # Flips preserve tree species and operate only during training.
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical", name="augmentation_flip")(inputs)

    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="species")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="treesat_cnn_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------- Main workflow --------------------

def main():
    global IMAGE_LOOKUP

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset split CSV not found:\n{DATASET_CSV}\n"
            "Run your existing dataset preparation scripts first."
        )

    df = pd.read_csv(DATASET_CSV)
    species_col = find_column(df, ["species", "class", "label", "tree_species"])
    split_col = find_column(df, ["split", "set", "subset", "partition"])
    filename_col = find_column(
        df, ["filename", "file_name", "image_filename", "image", "image_name", "path", "filepath", "image_path"]
    )
    path_col = find_column(df, ["filepath", "image_path", "full_path"], required=False)

    df[species_col] = df[species_col].astype(str).str.strip()
    df[split_col] = df[split_col].astype(str).str.strip().str.lower()
    df = df[df[species_col].isin(CLASS_NAMES)].copy()

    # Accept common names used in official split files.
    train_names = {"train", "training"}
    test_names = {"test", "testing"}
    official_train = df[df[split_col].isin(train_names)].copy()
    official_test = df[df[split_col].isin(test_names)].copy()

    if official_train.empty or official_test.empty:
        raise ValueError(
            f"Could not identify train/test rows from column '{split_col}'. "
            f"Values found: {sorted(df[split_col].unique())}"
        )

    print("Indexing TIFF files (this should only take a short time)...")
    IMAGE_LOOKUP = build_image_lookup()

    def attach_paths(part):
        part = part.copy()
        resolved = []
        missing = []
        for _, row in part.iterrows():
            path = resolve_image_path(row, path_col, filename_col, row[species_col])
            if path is None or not Path(path).exists():
                missing.append(str(row[filename_col]))
                resolved.append(None)
            else:
                resolved.append(str(path))
        part["resolved_path"] = resolved
        if missing:
            print(f"Warning: skipping {len(missing)} rows with missing TIFF files.")
            print("First missing examples:", missing[:5])
            part = part[part["resolved_path"].notna()].copy()
        return part

    official_train = attach_paths(official_train)
    official_test = attach_paths(official_test)

    if official_train.empty or official_test.empty:
        raise ValueError("No usable train or test images remain after path resolution.")

    # Create validation only from the official training split; leave official test untouched.
    train_df, val_df = train_test_split(
        official_train,
        test_size=0.15,
        random_state=SEED,
        stratify=official_train[species_col],
    )

    for part in (train_df, val_df, official_test):
        part["label_id"] = part[species_col].map({name: i for i, name in enumerate(CLASS_NAMES)}).astype(int)

    print("\nClasses:", CLASS_NAMES)
    print(f"Official training samples available: {len(official_train)}")
    print(f"Training: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Official test: {len(official_test)}")
    print("\nTraining samples by class:")
    print(train_df[species_col].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    test_ds = make_dataset(official_test, training=False)

    model = build_model()
    model.summary()

    best_model_path = MODELS_DIR / "cnn_v2_best.keras"
    final_model_path = MODELS_DIR / "cnn_v2_final.keras"
    history_path = RESULTS_DIR / "cnn_v2_training_history.csv"
    report_path = RESULTS_DIR / "cnn_v2_classification_report.txt"
    cm_csv_path = RESULTS_DIR / "cnn_v2_confusion_matrix.csv"
    cm_png_path = RESULTS_DIR / "cnn_v2_confusion_matrix.png"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print("\nStarting CNN V2 training on CPU...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        steps_per_epoch=math.ceil(len(train_df) / BATCH_SIZE),
        validation_steps=math.ceil(len(val_df) / BATCH_SIZE),
        callbacks=callbacks,
        verbose=1,
    )

    pd.DataFrame(history.history).to_csv(history_path, index=False)

    # Load the best checkpoint before the final test evaluation.
    if best_model_path.exists():
        model = tf.keras.models.load_model(best_model_path)

    print("\nEvaluating once on the untouched official test set...")
    probabilities = model.predict(
        test_ds,
        steps=math.ceil(len(official_test) / BATCH_SIZE),
        verbose=1,
    )
    y_pred = np.argmax(probabilities, axis=1)
    y_true = official_test["label_id"].to_numpy(dtype=np.int32)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    print(f"\nCNN V2 official test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification report:\n")
    print(report)
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(cm)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Official test accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)\n\n")
        file.write("Class order: " + ", ".join(CLASS_NAMES) + "\n\n")
        file.write(report)
        file.write("\n\nConfusion matrix (rows=true, columns=predicted):\n")
        file.write(np.array2string(cm))

    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(cm_csv_path)

    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"CNN V2 Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")
    fig.tight_layout()
    fig.savefig(cm_png_path, dpi=160)
    plt.close(fig)

    # Save a final copy distinct from the checkpoint and previous CNN.
    model.save(final_model_path)

    summary = {
        "model": "CNN V2",
        "classes": CLASS_NAMES,
        "official_test_samples": int(len(official_test)),
        "test_accuracy": float(accuracy),
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "seed": SEED,
        "test_split_source": str(DATASET_CSV),
        "best_model": str(best_model_path),
        "final_model": str(final_model_path),
        "classification_report": str(report_path),
        "confusion_matrix_csv": str(cm_csv_path),
        "confusion_matrix_png": str(cm_png_path),
        "training_history": str(history_path),
    }
    with open(RESULTS_DIR / "cnn_v2_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nSaved CNN V2 outputs:")
    for path in [
        best_model_path, final_model_path, history_path, report_path,
        cm_csv_path, cm_png_path, RESULTS_DIR / "cnn_v2_summary.json"
    ]:
        print(" -", path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
