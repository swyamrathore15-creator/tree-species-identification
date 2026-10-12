
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import random
import math

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

# ---------------- CONFIGURATION ----------------

SEED = 42
BATCH_SIZE = 8
EPOCHS = 25
PATIENCE = 5
LEARNING_RATE = 0.001
SHUFFLE_BUFFER = 256

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/processed/balanced_dataset_split.csv"
DATA_DIR = ROOT / "data/raw/TreeSatAI"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

CLASSES = ["Fagus", "Picea", "Pinus", "Quercus"]

FOLDERS = {
    "Fagus": "aerial_60m_fagus_sylvatica",
    "Picea": "aerial_60m_picea_abies",
    "Pinus": "aerial_60m_pinus_sylvestris",
    "Quercus": "aerial_60m_quercus_petraea",
}

IMAGE_SHAPE = (304, 304, 4)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Use a moderate number of CPU threads.
try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass


# ---------------- DATASET PREPARATION ----------------

def find_column(df, possible_names):
    columns = {str(c).strip().lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in columns:
            return columns[name.lower()]
    raise ValueError(
        f"Could not find one of {possible_names}. "
        f"Available columns: {list(df.columns)}"
    )


def build_image_lookup():
    lookup = {}

    for species, folder_name in FOLDERS.items():
        folder = DATA_DIR / folder_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for pattern in ("*.tif", "*.tiff"):
            for path in folder.rglob(pattern):
                lookup[path.name.lower()] = path

    return lookup


def resolve_path(row, filename_col, path_col, lookup):
    # Prefer a valid path if the CSV contains one.
    if path_col is not None and pd.notna(row[path_col]):
        candidate = Path(str(row[path_col]))

        options = [candidate]
        if not candidate.is_absolute():
            options += [ROOT / candidate, DATA_DIR / candidate]

        for option in options:
            if option.exists():
                return str(option)

    # Otherwise find the TIFF by filename.
    filename = Path(str(row[filename_col])).name
    return str(lookup[filename.lower()]) if filename.lower() in lookup else None


def load_image(path):
    # TreeSatAI aerial band order: NIR, Green, Blue, Red.
    with rasterio.open(path) as src:
        image = src.read()

    if image.shape != (4, 304, 304):
        raise ValueError(f"Unexpected TIFF shape {image.shape}: {path}")

    image = np.transpose(image, (1, 2, 0)).astype(np.float32)
    return image / 255.0


def make_dataset(frame, training=False, repeat=False):
    paths = frame["resolved_path"].to_numpy()
    labels = frame["label_id"].to_numpy(dtype=np.int32)

    def generator():
        for path, label in zip(paths, labels):
            yield load_image(path), np.int32(label)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(IMAGE_SHAPE, tf.float32),
            tf.TensorSpec((), tf.int32),
        ),
    )

    if training:
        ds = ds.shuffle(
            buffer_size=min(SHUFFLE_BUFFER, len(frame)),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    # Repeat train and validation so each requested epoch has enough batches.
    # The test dataset must remain finite.
    if repeat:
        ds = ds.repeat()

    return ds.prefetch(1)


# ---------------- CNN MODEL ----------------

def build_model():
    inputs = tf.keras.Input(shape=IMAGE_SHAPE)

    x = tf.keras.layers.RandomFlip(
        "horizontal_and_vertical"
    )(inputs)

    x = tf.keras.layers.Conv2D(
        16, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        32, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        64, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    outputs = tf.keras.layers.Dense(
        len(CLASSES), activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="treesat_cnn_v2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------- TRAINING ----------------

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    species_col = find_column(
        df, ["species", "class", "label", "tree_species"]
    )
    split_col = find_column(
        df, ["split", "set", "subset", "partition"]
    )
    filename_col = find_column(
        df, ["filename", "file_name", "image_filename",
             "image", "image_name", "path", "filepath", "image_path"]
    )

    path_col = next(
        (c for c in df.columns
         if str(c).strip().lower() in
         ["filepath", "image_path", "full_path"]),
        None,
    )

    df[species_col] = df[species_col].astype(str).str.strip()
    df[split_col] = df[split_col].astype(str).str.strip().str.lower()
    df = df[df[species_col].isin(CLASSES)].copy()

    official_train = df[
        df[split_col].isin(["train", "training"])
    ].copy()

    official_test = df[
        df[split_col].isin(["test", "testing"])
    ].copy()

    if official_train.empty or official_test.empty:
        raise ValueError(
            f"Cannot identify official train/test rows. "
            f"Split values: {df[split_col].unique()}"
        )

    print("Indexing TIFF files...")
    lookup = build_image_lookup()

    for part in (official_train, official_test):
        part["resolved_path"] = part.apply(
            lambda row: resolve_path(
                row, filename_col, path_col, lookup
            ),
            axis=1,
        )

    missing_train = official_train["resolved_path"].isna().sum()
    missing_test = official_test["resolved_path"].isna().sum()

    if missing_train or missing_test:
        print(
            f"Skipping missing files: train={missing_train}, "
            f"test={missing_test}"
        )

    official_train = official_train.dropna(
        subset=["resolved_path"]
    ).copy()
    official_test = official_test.dropna(
        subset=["resolved_path"]
    ).copy()

    train_df, val_df = train_test_split(
        official_train,
        test_size=0.15,
        random_state=SEED,
        stratify=official_train[species_col],
    )

    label_map = {name: i for i, name in enumerate(CLASSES)}

    for part in (train_df, val_df, official_test):
        part["label_id"] = (
            part[species_col].map(label_map).astype(int)
        )

    print("\nClasses:", CLASSES)
    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Official test:", len(official_test))

    train_ds = make_dataset(train_df, training=True, repeat=True)
    val_ds = make_dataset(val_df, repeat=True)
    test_ds = make_dataset(official_test, repeat=False)

    model = build_model()
    model.summary()

    best_path = MODEL_DIR / "cnn_v2_fixed_best.keras"
    final_path = MODEL_DIR / "cnn_v2_fixed_final.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path),
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

    print("\nStarting fixed CNN V2 training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=math.ceil(len(train_df) / BATCH_SIZE),
        validation_steps=math.ceil(len(val_df) / BATCH_SIZE),
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )

    pd.DataFrame(history.history).to_csv(
        RESULT_DIR / "cnn_v2_fixed_history.csv", index=False
    )

    # Evaluate the best checkpoint, not the last epoch.
    model = tf.keras.models.load_model(best_path)

    print("\nEvaluating official test set...")
    probabilities = model.predict(
        test_ds,
        steps=math.ceil(len(official_test) / BATCH_SIZE),
        verbose=1,
    )

    y_true = official_test["label_id"].to_numpy(dtype=np.int32)
    y_pred = np.argmax(probabilities, axis=1)

    # Defensive check: predictions and labels must match one-to-one.
    if len(y_pred) != len(y_true):
        raise RuntimeError(
            f"Prediction count {len(y_pred)} does not match "
            f"test label count {len(y_true)}."
        )

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASSES)))
    )

    print(f"\nCNN V2 fixed test accuracy: {accuracy:.4f} "
          f"({accuracy * 100:.2f}%)")
    print("\nClassification report:\n")
    print(report)
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(cm)

    with open(
        RESULT_DIR / "cnn_v2_fixed_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Test accuracy: {accuracy:.6f}\n\n")
        file.write("Class order: " + ", ".join(CLASSES) + "\n\n")
        file.write(report)
        file.write("\n\nConfusion matrix:\n")
        file.write(np.array2string(cm))

    pd.DataFrame(
        cm, index=CLASSES, columns=CLASSES
    ).to_csv(RESULT_DIR / "cnn_v2_fixed_confusion_matrix.csv")

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=CLASSES
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"CNN V2 Fixed - Accuracy {accuracy * 100:.2f}%")
    fig.tight_layout()
    fig.savefig(
        RESULT_DIR / "cnn_v2_fixed_confusion_matrix.png", dpi=160
    )
    plt.close(fig)

    model.save(final_path)

    with open(
        RESULT_DIR / "cnn_v2_fixed_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "accuracy": float(accuracy),
                "test_samples": int(len(y_true)),
                "classes": CLASSES,
                "best_model": str(best_path),
                "final_model": str(final_path),
            },
            file,
            indent=2,
        )

    print("\nSaved model and results:")
    print(best_path)
    print(final_path)
    print(RESULT_DIR / "cnn_v2_fixed_report.txt")
    print(RESULT_DIR / "cnn_v2_fixed_confusion_matrix.csv")
    print(RESULT_DIR / "cnn_v2_fixed_confusion_matrix.png")
    print(RESULT_DIR / "cnn_v2_fixed_history.csv")


if __name__ == "__main__":
    main()
