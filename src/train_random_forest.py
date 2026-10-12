
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Paths
DATA_FILE = (
    r"D:\ML PROJECT\Tree Species Identification"
    r"\data\processed\image_features.csv"
)

MODEL_DIR = (
    r"D:\ML PROJECT\Tree Species Identification\models"
)

RESULT_DIR = (
    r"D:\ML PROJECT\Tree Species Identification\results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Load extracted features
df = pd.read_csv(DATA_FILE)

# Use only training rows to fit the model
train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

# Metadata columns are not model features
excluded = {"filename", "species", "split"}
feature_columns = [
    column for column in df.columns
    if column not in excluded
]

X_train = train_df[feature_columns]
y_train = train_df["species"]

X_test = test_df[feature_columns]
y_test = test_df["species"]

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Number of features:", len(feature_columns))

# Random Forest does not require feature scaling.
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Evaluate on official test samples
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

print(f"Exact Random Forest Macro F1: {macro_f1:.4f}")
print(f"Exact Random Forest Macro F1 (%): {macro_f1 * 100:.2f}%")

print("\n" + "=" * 55)
print("RANDOM FOREST RESULTS")
print("=" * 55)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification report:")
report = classification_report(
    y_test,
    predictions,
    zero_division=0,
)
print(report)

print("\nConfusion matrix:")
labels = sorted(df["species"].unique())
cm = confusion_matrix(y_test, predictions, labels=labels)
print("Class order:", labels)
print(cm)

# Save model and evaluation artifacts
joblib.dump(
    {"model": model, "features": feature_columns},
    os.path.join(MODEL_DIR, "random_forest.joblib"),
)

pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
).to_csv(os.path.join(RESULT_DIR, "rf_confusion_matrix.csv"))

with open(
    os.path.join(RESULT_DIR, "rf_classification_report.txt"),
    "w",
    encoding="utf-8",
) as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("\nSaved model to models/random_forest.joblib")
print("Saved evaluation files to results/")
