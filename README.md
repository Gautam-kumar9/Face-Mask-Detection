# MaskSense AI: Face Mask Detection Web App

MaskSense AI is a Streamlit-based computer vision application that classifies whether a person is wearing a face mask. The project is designed for fast demos, portfolio presentation, and cloud deployment.

## Overview

The app uses a TensorFlow/Keras model and provides two inference modes:

- Image upload for quick testing
- Browser camera capture for real-time interaction

Predictions are presented with class confidence to make outputs easy to interpret.

## Key Features

- Professional Streamlit interface with custom styling
- Dual input pipeline: upload and camera capture
- Lightweight inference workflow optimized for web usage
- Streamlit Community Cloud-ready configuration
- Clean, code-only repository structure for fast GitHub pushes

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow

## Repository Structure

```text
face-mask-detector/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- runtime.txt
|-- .streamlit/
|   `-- config.toml
`-- README.md
```

## Quick Start (Local)

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your trained model file in the project root:

```text
mask_detector_model.h5
```

5. Run the app:

```bash
streamlit run app.py
```

## Model File Note

To keep repository size small and push operations fast, large assets are excluded from version control. This includes:

- Dataset directory (`data/`)
- Trained model files (`*.h5`)

Before running or deploying, make sure `mask_detector_model.h5` exists in the project root.

For cloud deployment without committing large model files, the app can automatically download the model from a URL using `MODEL_URL`.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open Streamlit Community Cloud and click **Create app**.
3. Select repository and branch.
4. Set main file path to `app.py`.
5. In **Advanced settings -> Secrets**, add:

```toml
MODEL_URL = "https://your-public-model-url/mask_detector_model.h5"
```

6. Deploy.

## Training (Optional)

If you need to retrain the model:

```bash
python train_model.py
```

After training, copy or move the generated `mask_detector_model.h5` to the project root used by `app.py`.

## Future Improvements

- Add face detection before mask classification
- Improve model performance with transfer learning
- Add experiment tracking and evaluation metrics
- Add CI checks for linting and dependency validation

## Author

Gautam Kumar
