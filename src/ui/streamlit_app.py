import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os

API_URL = "http://localhost:8000/predict"  # docker network name


st.set_page_config(page_title="Monkey Classifier 🐒", layout="wide")

st.title("🐒 Monkey Classification & Similarity Search")

uploaded_file = st.file_uploader(
    "Upload a monkey image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    if st.button("Classify"):
        with st.spinner("Thinking like a monkey expert..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(API_URL, files=files)

        if response.status_code != 200:
            st.error("API error 😢")
        else:
            data = response.json()

            st.success(f"Predicted class: {data['predicted_class']}")

            st.subheader("Similar images")
            cols = st.columns(5)

            for col, img_path in zip(cols, data["similar_images"]):
                try:
                    img = Image.open(img_path)
                    col.image(img, use_column_width=True)
                except Exception:
                    col.write("⚠️ not found")