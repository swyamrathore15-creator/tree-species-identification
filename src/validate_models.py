
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "TreeSatAI"
FEATURES_FILE = ROOT / "data" / "processed" / "image_features.csv"
RF_FILE = ROOT / "models" / "random_forest.joblib"
CNN_FILE = ROOT / "models" / "cnn_v2_fixed_best.keras"
RESULT_DIR = ROOT / "results"

CLASSES = ["Fagus", "Picea", "Pinus", "Quercus"]

FOLDERS = {
    "Fagus": "aerial_60m_fagus_sylvatica",
    "Picea": "aerial_60m_picea_abies",
    "Pinus": "aerial_60m_pinus_sylvestris",
    "Quercus": "aerial_60m_quercus_petraea",
}


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the same test feature records used for RF evaluation.
    df = pd.read_csv(FEATURES_FILE)
    test_df = df[
        (df["split"].astype(str).str.lower() == "test")
        & df["species"].isin(CLASSES)
    ].copy().reset_index(drop=True)

    if test_df.empty:
        raise ValueError("No official test rows found in image_features.csv")

    saved_rf = joblib.load(RF_FILE)
    rf = saved_rf["model"]
    feature_columns = saved_rf["features"]

    missing = [c for c in feature_columns if c not in test_df.columns]
    if missing:
        raise ValueError(f"Missing Random Forest features: {missing}")

    X_test = test_df[feature_columns]
    y_true = test_df["species"].to_numpy()

    # Load the saved CNN.
    cnn = tf.keras.models.load_model(CNN_FILE, compile=False)

    # Build a filename-to-path lookup for the four model classes.
    lookup = {}
    for species, folder_name in FOLDERS.items():
        folder = DATA_DIR / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing species folder: {folder}")

        for pattern in ("*.tif", "*.tiff"):
            for path in folder.rglob(pattern):
                lookup[path.name.lower()] = path

    # Confirm every test image can be located.
    paths = []
    for filename in test_df["filename"]:
        key = Path(str(filename)).name.lower()
        if key not in lookup:
            raise FileNotFoundError(f"Cannot find test TIFF: {filename}")
        paths.append(lookup[key])

    print("Official test images:", len(test_df))
    print("Loading CNN images in small batches...")

    cnn_predictions = []
    cnn_probabilities = []
    batch_size = 8

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images = []

        for path in batch_paths:
            with rasterio.open(path) as src:
                image = src.read()

            if image.shape != (4, 304, 304):
                raise ValueError(
                    f"Unexpected image shape {image.shape}: {path.name}"
                )

            # Match training: NIR, Green, Blue, Red;
            # H x W x channels; float32 normalized by 255.
            image = np.transpose(image, (1, 2, 0)).astype(np.float32)
            images.append(image / 255.0)

        probabilities = cnn.predict(
            np.stack(images), verbose=0
        )

        cnn_probabilities.extend(probabilities.tolist())
        cnn_predictions.extend(
            np.argmax(probabilities, axis=1).tolist()
        )

        print(f"Processed {min(start + batch_size, len(paths))}/{len(paths)}")

    cnn_probabilities = np.asarray(cnn_probabilities)
    cnn_predictions = np.asarray(cnn_predictions)
    cnn_labels = np.asarray(CLASSES)[cnn_predictions]

    # Random Forest predictions.
    rf_predictions = rf.predict(X_test)
    rf_probabilities = rf.predict_proba(X_test)
    rf_classes = list(rf.classes_)

    # Calculate metrics and save reports.
    results = []
    predictions_df = pd.DataFrame({
        "filename": test_df["filename"],
        "actual_species": y_true,
        "random_forest_prediction": rf_predictions,
        "cnn_prediction": cnn_labels,
    })

    for name, predictions in [
        ("Random Forest", rf_predictions),
        ("CNN V2 Fixed", cnn_labels),
    ]:
        accuracy = accuracy_score(y_true, predictions)
        macro_f1 = f1_score(
            y_true, predictions, labels=CLASSES,
            average="macro", zero_division=0
        )

        print(f"\n{'=' * 50}")
        print(name)
        print(f"Accuracy: {accuracy:.4%}")
        print(f"Macro F1: {macro_f1:.4%}")
        print(classification_report(
            y_true, predictions,
            labels=CLASSES, digits=4, zero_division=0
        ))

        cm = confusion_matrix(y_true, predictions, labels=CLASSES)
        print("Confusion matrix (rows=true, columns=predicted):")
        print(cm)

        safe_name = name.lower().replace(" ", "_")
        pd.DataFrame(
            cm, index=CLASSES, columns=CLASSES
        ).to_csv(RESULT_DIR / f"{safe_name}_validation_confusion_matrix.csv")

        results.append({
            "Model": name,
            "Test samples": len(y_true),
            "Accuracy (%)": round(accuracy * 100, 2),
            "Macro F1-score (%)": round(macro_f1 * 100, 2),
        })

    # Save per-image results and summary.
    predictions_df.to_csv(
        RESULT_DIR / "batch_validation_predictions.csv", index=False
    )
    pd.DataFrame(results).to_csv(
        RESULT_DIR / "batch_validation_summary.csv", index=False
    )

    print("\nSaved:")
    print("results/batch_validation_predictions.csv")
    print("results/batch_validation_summary.csv")
    print("Individual model confusion matrices saved in results/")

    print("\nCompare these metrics with your original training reports.")
    print("If they differ, investigate test-row alignment and preprocessing.")


if __name__ == "__main__":
    main()
