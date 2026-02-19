import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Smart City Complaint Classifier")

st.title("🏙 Smart City Complaint Classifier")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_custom_model_name.h5")

model = load_model()

# IMPORTANT: Change class names according to your model
class_names = ["Garbage", "Roads", "Water", "Electricity"]

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
