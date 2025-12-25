import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="AI Defect Detector", page_icon="🔍")
st.title("🔍 AI Defect Detector")
st.write("Detects 6 defects: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches")

model = YOLO("/content/drive/MyDrive/AI-Defect-Detector/best.pt")

img = st.camera_input("📷 Take a photo")

if img is not None:
    image = Image.open(img).convert("RGB")
    st.image(image, caption="Captured Image")

    if st.button("🚀 Detect Defects"):
        results = model(image, conf=0.3)
        st.image(results[0].plot(), caption="Detected Defects")
