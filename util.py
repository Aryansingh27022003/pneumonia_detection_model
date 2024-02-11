import base64
import numpy as np
import streamlit as st
from PIL import ImageOps, Image

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    # Convert image to (224,224)
    image = ImageOps.fit(image, (224, 224), Image.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    # index = np.argmax(prediction)
    index=1 if prediction[0][1]>0.95 else 0
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print(prediction)

    return class_name, confidence_score
