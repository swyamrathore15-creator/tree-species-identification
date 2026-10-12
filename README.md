# Tree Species Identification Using UAV Multispectral Images

A machine learning project that classifies tree species from four-band multispectral aerial imagery using Random Forest and Convolutional Neural Network (CNN) models.

## Project Overview

This project explores automated tree species identification using multispectral imagery and machine learning. It uses the TreeSatAI dataset to classify images into four tree species.

The project includes dataset preparation, multispectral feature extraction, model training, evaluation, and a Streamlit web application for predictions.

**Note:** The project title references UAV imagery, but the dataset used is TreeSatAI aerial imagery. The images should not be described as personally collected UAV data.

## Supported Tree Species

- Fagus — European beech
- Picea — Norway spruce
- Pinus — Scots pine
- Quercus — Sessile oak

## Models

### Random Forest
Uses 25 features derived from multispectral band statistics and vegetation indices.

### CNN V2 Fixed
Uses a convolutional neural network to learn image features directly from four-band multispectral images.

## Dataset and Image Format

The project uses the TreeSatAI dataset.

Expected input for the current prediction application:

- File format: GeoTIFF (`.tif` or `.tiff`)
- Image dimensions: 304 × 304 pixels
- Number of bands: 4
- Band order: NIR, Green, Blue, Red
- CNN normalization: pixel values divided by 255

Ordinary three-band RGB photographs are not supported by the current application.

The dataset is not included in this repository. Obtain it from an authorized dataset source and place the files in the directory structure expected by the project scripts.

## Evaluation Results

Both models were evaluated on the same official test set of 308 images.

| Metric | Random Forest | CNN V2 Fixed |
|---|---:|---:|
| Test accuracy | 82.79% | 83.77% |
| Macro F1-score | 82.58% | 83.49% |
| Correct predictions | 255/308 | 258/308 |

CNN V2 Fixed performed slightly better on this test set. The difference is small, and these results do not establish that the CNN will outperform Random Forest on every dataset or deployment scenario.

## Project Structure

```text
Tree Species Identification/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── models/
├── notebooks/
├── results/
└── src/
    ├── extract_features.py
    ├── train_random_forest.py
    ├── train_cnn_v2_fixed.py
    ├── validate_models.py
    ├── compare_models.py
    └── plot_cnn_history.py
```

The raw dataset, processed dataset files, virtual environment, and trained model binaries are excluded from Git by default. The saved model files are required to run predictions and must be obtained or generated separately.

## Installation

Python 3.12 was used in the development environment.

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment.

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

4. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

5. Place the required saved model files in the `models/` directory:

- `random_forest.joblib`
- `cnn_v2_fixed_best.keras`

6. Ensure the dataset is available if you intend to run training or batch validation scripts.

## Run the Streamlit Application

From the project root, run:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, typically `http://localhost:8501`.

Upload a compatible four-band TIFF, select a model, and click **Identify tree species**.

## Run Batch Validation

After placing the dataset, processed features, and saved models in the expected locations, run:

```bash
python src/validate_models.py
```

Validation summaries, per-image predictions, and confusion matrices are saved in the `results/` directory.

## Limitations

- Predictions are limited to the four supported species.
- The current application expects a specific four-band image format and band order.
- Model confidence is not a guarantee of classification correctness.
- Performance may vary across geographical regions, image acquisition conditions, and datasets.
- The manually tested examples are sanity checks; the reported evaluation metrics come from the official held-out test set.

## Future Improvements

- Add more species and diverse imagery.
- Improve the interface with multispectral band visualizations.
- Evaluate generalization across different acquisition conditions.
- Explore model optimization and deployment options.

## Technologies Used

Python, TensorFlow/Keras, Scikit-learn, Rasterio, NumPy, Pandas, Matplotlib, and Streamlit.

## Academic Project

Tree Species Identification Using UAV Multispectral Images.
