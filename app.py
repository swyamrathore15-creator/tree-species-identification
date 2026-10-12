
from pathlib import Path

import joblib
import numpy as np
import rasterio
import streamlit as st
import tensorflow as tf

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

RF_PATH = MODEL_DIR / "random_forest.joblib"
CNN_PATH = MODEL_DIR / "cnn_v2_fixed_best.keras"

CLASSES = ["Fagus", "Picea", "Pinus", "Quercus"]
IMAGE_SHAPE = (304, 304, 4)


# --------------------------------------------------
# Load models once and reuse them
# --------------------------------------------------
@st.cache_resource
def load_random_forest():
    saved = joblib.load(RF_PATH)
    return saved["model"], saved["features"]


@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model(CNN_PATH, compile=False)


# --------------------------------------------------
# Image loading and preprocessing
# Band order: NIR, Green, Blue, Red
# --------------------------------------------------
def read_multispectral_image(uploaded_file):
    uploaded_file.seek(0)

    with rasterio.open(uploaded_file) as src:
        image = src.read()

    if image.shape != (4, 304, 304):
        raise ValueError(
            "Expected a 4-band TIFF with dimensions "
            "304 x 304 pixels. "
            f"Received array shape: {image.shape}"
        )

    # CNN input: H x W x C, normalized to 0-1
    cnn_image = np.transpose(image, (1, 2, 0)).astype(
        np.float32
    )
    cnn_image /= 255.0

    return image.astype(np.float32), cnn_image


# --------------------------------------------------
# Random Forest feature extraction
# Must match src/extract_features.py
# --------------------------------------------------
def band_stats(values, prefix):
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def extract_rf_features(image):
    nir, green, blue, red = image

    features = {}

    for name, band in [
        ("nir", nir),
        ("green", green),
        ("blue", blue),
        ("red", red),
    ]:
        features.update(band_stats(band / 255.0, name))

    denominator = nir + red
    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(nir),
        where=denominator != 0,
    )

    green_denominator = nir + green
    nir_green_index = np.divide(
        nir - green,
        green_denominator,
        out=np.zeros_like(nir),
        where=green_denominator != 0,
    )

    features.update(band_stats(ndvi, "ndvi"))
    features.update(
        band_stats(nir_green_index, "nir_green_index")
    )

    features["zero_ndvi_denominator_fraction"] = float(
        np.mean(denominator == 0)
    )

    return features


# --------------------------------------------------
# User interface
# --------------------------------------------------
st.set_page_config(
    page_title="Tree Species Identifier",
    page_icon="🌳",
    layout="centered",
)

st.title("🌳 Tree Species Identification")
st.write(
    "Identify tree species using four-band multispectral "
    "aerial imagery."
)

st.info(
    "Upload a compatible 4-band TIFF: 304 × 304 pixels, "
    "with bands ordered NIR, Green, Blue, Red. "
    "Ordinary RGB photographs are not supported."
)

uploaded_file = st.file_uploader(
    "Upload multispectral TIFF",
    type=["tif", "tiff"],
)

model_choice = st.selectbox(
    "Choose a classification model",
    ["CNN V2 Fixed", "Random Forest"],
)

if uploaded_file is not None:
    st.write(f"**File:** {uploaded_file.name}")

    if st.button("Identify tree species", type="primary"):
        try:
            raw_image, cnn_image = read_multispectral_image(
                uploaded_file
            )

            with st.spinner("Analyzing multispectral image..."):
                if model_choice == "CNN V2 Fixed":
                    model = load_cnn()
                    probabilities = model.predict(
                        np.expand_dims(cnn_image, axis=0),
                        verbose=0,
                    )[0]

                else:
                    model, feature_columns = load_random_forest()
                    feature_values = extract_rf_features(raw_image)

                    missing = [
                        name for name in feature_columns
                        if name not in feature_values
                    ]
                    if missing:
                        raise ValueError(
                            f"Missing model features: {missing}"
                        )

                    row = np.array(
                        [feature_values[name]
                         for name in feature_columns],
                        dtype=np.float32,
                    ).reshape(1, -1)

                    probabilities = model.predict_proba(row)[0]

                    # Map model probability columns to class names.
                    class_order = list(model.classes_)
                    probabilities = np.array([
                        probabilities[class_order.index(name)]
                        if name in class_order else 0.0
                        for name in CLASSES
                    ])

            predicted_index = int(np.argmax(probabilities))
            predicted_species = CLASSES[predicted_index]
            confidence = float(probabilities[predicted_index])

            st.success(f"Predicted species: {predicted_species}")
            st.metric("Model confidence", f"{confidence:.1%}")

            st.subheader("Class probabilities")
            st.bar_chart({
                "Probability": {
                    name: float(probabilities[i])
                    for i, name in enumerate(CLASSES)
                }
            })

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee that the classification is correct."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
