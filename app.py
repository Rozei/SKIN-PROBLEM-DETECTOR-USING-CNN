import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import tempfile
from PIL import Image

# Load model
model = load_model("skin_problem_model.h5")
labels = ['dark_circles', 'enlarged_pores', 'fine_lines', 'hyperpigmentation', 'pimples', 'wrinkles']

# Load medication suggestions
with open("medication.json", "r") as f:
    medication_data = json.load(f)

st.set_page_config(page_title="Facial Skin Analyzer", layout="centered")
st.title("💆‍♂️ Facial Skin Problem Detector")
st.write("Click below to capture a photo using your webcam and detect skin issues.")

# Camera input
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert to CV2 image
    img = Image.open(img_file_buffer)
    img = img.resize((224, 224))  # Match model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)[0]

    st.subheader("🧪 Prediction Results")
    for i, label in enumerate(labels):
        prob = preds[i] * 100
        st.write(f"**{label.replace('_', ' ').title()}**: {prob:.2f}%")

        # Show basic suggestions for high probability
        if prob > 50:
            st.info(f"💡 **Tip for {label.replace('_', ' ').title()}**: {medication_data.get(label, 'No data available.')}")
