import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("pneumonia_model.h5")

# Define prediction function
def predict_pneumonia(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Pneumonia" if prediction > 0.5 else "Normal"

# UI
st.title("🫁 Pneumonia Detector")
st.write("Upload a chest X-ray image and the model will predict if it shows signs of pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Analyzing..."):
        result = predict_pneumonia(img)
    st.success(f"Prediction: **{result}**")

    st.write("Note: This model is for educational purposes only and should not be used for medical diagnosis.")