import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("Weather_App2.hdf5")  # Adjust path as necessary
    return model

model = load_model()

st.write("""
# Weather Image Classification System
""")

file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, method=Image.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Use COLOR_RGB2BGR instead
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
        string = "OUTPUT: " + class_names[np.argmax(prediction)]
        st.success(string)
    except Exception as e:
        st.error(f"An error occurred: {e}")
