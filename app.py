import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path
from pathlib import Path
from tensorflow.keras.models import load_model

model_load = load_model('model')

st.title('CIFAR10 Image Recognizer')
labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
# st.header(":white[Sample images for classes]")

# clas = st.radio(
# "Choose class",
# classes, horizontal=True)
# images = get_images(option, clas)
# rand = randint(0, 9)
# a = cv2.resize(images[rand], (128,128), interpolation = cv2.INTER_AREA)
# st.image(a)
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)


if st.button('Predict'):
    try:
        img_array = cv2.resize(img_array.astype('uint8'), (32, 32))
        img_array = np.expand_dims(img_array, axis=1)
        img_array = img_array.transpose((1,0,2,3))
        val = model_load.predict(img_array)
        st.write(f'result: {classes[np.argmax(val[0])]}')
#         st.bar_chart(val[0])
    except:
        pass
