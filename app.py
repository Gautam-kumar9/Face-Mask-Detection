# app.py
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="MaskSense AI",
    page_icon="😷",
    layout="wide",
)

MODEL_PATH = Path(__file__).resolve().parent / "mask_detector_model.h5"


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found: mask_detector_model.h5")
        st.stop()
    return load_model(str(MODEL_PATH), compile=False)


def predict_mask(model, image_np):
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]

    resized = cv2.resize(image_np, (128, 128), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    batch = np.expand_dims(normalized, axis=0)

    without_mask_prob = float(model.predict(batch, verbose=0)[0][0])
    with_mask_prob = 1.0 - without_mask_prob

    if with_mask_prob >= without_mask_prob:
        return "With Mask", with_mask_prob, with_mask_prob, without_mask_prob
    return "Without Mask", without_mask_prob, with_mask_prob, without_mask_prob


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Outfit:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 85% -10%, #b8f2e6 10%, rgba(184, 242, 230, 0) 60%),
            radial-gradient(900px 500px at -10% 10%, #f7d6e0 10%, rgba(247, 214, 224, 0) 55%),
            linear-gradient(135deg, #f7f9fc 0%, #eef4ff 100%);
    }
    .hero {
        border-radius: 18px;
        padding: 1.3rem 1.2rem;
        background: linear-gradient(120deg, rgba(23, 43, 77, 0.9), rgba(48, 78, 144, 0.92));
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(26, 48, 96, 0.25);
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        margin: 0;
        letter-spacing: 0.2px;
        font-size: 2rem;
    }
    .hero p {
        margin-top: 0.4rem;
        opacity: 0.92;
    }
    .glass-card {
        border-radius: 16px;
        padding: 0.85rem 1rem;
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.7);
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>MaskSense AI</h1>
      <p>Fast face-mask classification with confidence insights for image upload and live camera capture.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = get_model()

st.sidebar.header("Project Info")
st.sidebar.write("Built with Streamlit, TensorFlow, OpenCV, and PIL.")
st.sidebar.info("Tip: For best results, use a front-facing face image with good lighting.")

input_mode = st.radio(
    "Select Input Source",
    ("Upload Image", "Camera Capture"),
    horizontal=True,
)


def show_prediction(image_obj):
    image_rgb = image_obj.convert("RGB")
    image_np = np.array(image_rgb)
    label, confidence, with_mask_prob, without_mask_prob = predict_mask(model, image_np)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image(image_rgb, caption="Input Image", width="stretch")

    with col2:
        state = "success" if label == "With Mask" else "error"
        message = f"Prediction: {label} ({confidence * 100:.2f}%)"
        if state == "success":
            st.success(message)
        else:
            st.error(message)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.metric("With Mask Probability", f"{with_mask_prob * 100:.2f}%")
        st.progress(min(max(with_mask_prob, 0.0), 1.0))
        st.metric("Without Mask Probability", f"{without_mask_prob * 100:.2f}%")
        st.progress(min(max(without_mask_prob, 0.0), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)


if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        show_prediction(uploaded_image)
else:
    camera_file = st.camera_input("Capture image using your camera")
    if camera_file is not None:
        camera_image = Image.open(camera_file)
        show_prediction(camera_image)

with st.expander("How to use"):
    st.write("1. Choose Upload Image or Camera Capture.")
    st.write("2. Provide a clear face image.")
    st.write("3. Review label and confidence bars.")
