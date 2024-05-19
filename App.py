import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
# Load your trained model (replace 'model' with your actual model)
# Example: model = load_model('path/to/your/model.h5')
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("Weather_App_Finals.keras")  # Adjust path as necessary
    return model

model = load_model()

st.set_page_config(page_title="Weather Image Classification", page_icon="🌦️")
st.write("""
# Weather Image Classification System
""")
st.write(f'Allen Gerald G. Opeña')
st.write(f'CPE32S1  May 19, 2024')
file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "png"])


def import_and_predict(image_data, model):
    size = (150, 150)  # Adjust size as per model's requirement
    image = ImageOps.fit(image_data, size)
    img_array = np.asarray(image)
    img_array = img_array[np.newaxis, ...]  # Create a batch of one image
    img_array = img_array / 255.0  # Normalize the image

    prediction = model.predict(img_array)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    confidence_percentage = round(np.max(prediction) * 100, 2)
    st.success(f"OUTPUT: {predicted_class_label}")
    st.write(f"Confidence: {confidence_percentage}%")
     
# Example images
example_images = {
    'Cloudy':'Cloudy.jpg',
    'Rain':'Rain.jpg',
    'Shine':'Shine.jpg',
    'Sunrise':'Sunrise.jpg'
}
# Displaying example images for each category
st.write("## Example of Weather Images")
for label, path in example_images.items():
    image = Image.open(path)
    st.image(image, caption=f'Example of {label}', use_column_width=True)
    
