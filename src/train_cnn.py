
import os
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

CSV_PATH = (
    r"D:\ML PROJECT\Tree Species Identification"
    r"\data\processed\balanced_dataset_split.csv"
)

MODEL_DIR = (
    r"D:\ML PROJECT\Tree Species Identification\models"
)

RESULT_DIR = (
    r"D:\ML PROJECT\Tree Species Identification\results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

IMAGE_SIZE = 304
BATCH_SIZE = 8
EPOCHS = 20
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# --------------------------------------------------
# LOAD DATASET AND SPLITS
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

classes = sorted(df["species"].unique())
class_to_id = {name: i for i, name in enumerate(classes)}

print("Classes:", classes)
print("Official training samples:", len(train_df))
print("Official test samples:", len(test_df))

# Validation split comes only from official training data
fit_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["species"],
)

print("Training:", len(fit_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# --------------------------------------------------
# IMAGE LOADER
# Band order: NIR, Green, Blue, Red
# --------------------------------------------------

def load_image(path):
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)

    if image.shape != (4, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Unexpected image shape {image.shape}: {path}"
        )

    # Convert (bands, height, width) to
    # (height, width, bands), normalize to 0-1
    image = np.transpose(image, (1, 2, 0)) / 255.0

    return image


def make_dataset(frame, training=False):
    paths = frame["image_path"].tolist()
    labels = np.array(
        [class_to_id[s] for s in frame["species"]],
        dtype=np.int32,
    )

    def generator():
        for path, label in zip(paths, labels):
            yield load_image(path), label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(IMAGE_SIZE, IMAGE_SIZE, 4),
                dtype=tf.float32,
            ),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    if training:
        dataset = dataset.shuffle(
            min(len(frame), 512),
            seed=SEED,
        )

    return dataset.batch(BATCH_SIZE).prefetch(1)


train_ds = make_dataset(fit_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# --------------------------------------------------
# SMALL CNN FOR CPU TRAINING
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 4)
    ),

    tf.keras.layers.Conv2D(
        16, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        32, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(classes), activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_cnn.keras"),
        monitor="val_loss",
        save_best_only=True,
    ),
]

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# --------------------------------------------------
# FINAL OFFICIAL TEST EVALUATION
# --------------------------------------------------

print("\nEvaluating on official test set...")

probabilities = model.predict(test_ds)
predictions = np.argmax(probabilities, axis=1)

y_test = np.array(
    [class_to_id[s] for s in test_df["species"]],
    dtype=np.int32,
)

accuracy = accuracy_score(y_test, predictions)

print("\nCNN accuracy:", f"{accuracy:.4f}")
print("\nClassification report:")
report = classification_report(
    y_test,
    predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0,
)
print(report)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=list(range(len(classes))),
)

print("\nConfusion matrix (class order:", classes, ")")
print(cm)

model.save(os.path.join(MODEL_DIR, "final_cnn.keras"))

pd.DataFrame(
    cm,
    index=classes,
    columns=classes,
).to_csv(os.path.join(RESULT_DIR, "cnn_confusion_matrix.csv"))

with open(
    os.path.join(RESULT_DIR, "cnn_classification_report.txt"),
    "w",
    encoding="utf-8",
) as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("\nSaved CNN model and evaluation results.")
