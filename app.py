import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model (make sure the path is correct)
model = tf.keras.models.load_model('./model_lspds_roy.h5')

def classify_image(file):
    img = Image.open(file)
    img = img.convert('RGB')  # Ensure image is in RGB mode
    img = img.resize((200, 200))  # Resize to match the input size expected by the model
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image

    print("Image array shape:", img_array.shape)  # Debugging statement
    print("Image array data type:", img_array.dtype)  # Debugging statement

    # Make prediction
    prediction = model.predict(img_array)
    confidence_real = prediction[0][0]
    confidence_fake = 1 - confidence_real
    
    # Determine the label based on the prediction
    label = 'fake' if confidence_real < 0.5 else 'real'
    
    return {'label': label, 'confidence_real': confidence_real, 'confidence_fake': confidence_fake}

st.set_page_config(
    page_title="The Deep Roy",
    page_icon="🍆",
    layout="wide")

st.title("Upload an image to see if it is a deepfake or a real.")

st.image('./ImageBanner.gif', use_column_width=True)
st.write("*Video: [Lil beast](https://youtu.be/Ky0nwzlZrMk)*")
st.write("*Logo made using [BlueArchive-Style Logo Generator](https://tmp.nulla.top/ba-logo/)*")
st.subheader("Upload an image to see if it is a fake or real face:")
file_uploaded = st.file_uploader("Select Image File", type=["jpg", "png", "jpeg"])
if file_uploaded is not None:
    res = classify_image(file_uploaded)

    c1, buff, c2 = st.columns([2, 0.5, 2])

    c1.subheader("Uploaded Image")
    c1.image(file_uploaded, use_column_width=True)
    c2.subheader("Classification Result")
    c2.write("This image is classified as **{}**.".format(res['label'].title()))
    c2.write("Confidence Real: {:.2f}".format(res['confidence_real']))
    c2.write("Confidence Fake: {:.2f}".format(res['confidence_fake']))