# MaskSense AI - Face Mask Detector

A polished Streamlit web app that detects whether a person is wearing a face mask using a trained TensorFlow model.

## Features

- Modern and responsive Streamlit UI
- Upload image prediction
- Camera capture prediction (`st.camera_input`) for browser-friendly live capture
- Confidence bars for both classes
- Deploy-ready project structure for Streamlit Community Cloud

## Project Structure

- `app.py` - Streamlit application
- `train_model.py` - Model training script
- `mask_detector_model.h5` - Trained model file
- `requirements.txt` - Python dependencies for deployment
- `runtime.txt` - Python version for Streamlit Cloud
- `.streamlit/config.toml` - Theme config

## Run Locally

1. Create/activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud and click **Create app**.
3. Select your repo and branch.
4. Set main file path to `app.py`.
5. Deploy.

## Push to GitHub

Run from the project folder:

```bash
git init
git add .
git commit -m "Initial MaskSense AI app"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Notes

- Keep `mask_detector_model.h5` in the project root so `app.py` can load it.
- For retraining, run `train_model.py` and replace the model file.
