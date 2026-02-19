import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Smart City Complaint Classifier")
st.title("🏙 Smart City Complaint Classifier")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("smart_city_model.h5")
    return model

model = load_model()

# IMPORTANT: Replace with your actual class order
class_names = ['Garbage', 'Roads', 'Water', 'Electricity']

uploaded_file = st.file_uploader("Upload Complaint Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)

    st.success(f"Predicted Department: {class_names[index]}")
