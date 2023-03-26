import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from svgpathtools import parse_path
from pathlib import Path
from glob import glob
from tensorflow.keras.models import load_model

def get_images(clas):
    return [cv2.imread(i) for i in glob('images/'+clas+'*')]

model_load = load_model('model')

st.title('CIFAR10 Image Recognizer')
labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
clas = st.radio(
"Choose class",
labels, horizontal=True)
images = get_images(clas)
rand = randint(0, 9)
a = cv2.resize(images[rand], (128,128), interpolation = cv2.INTER_AREA)
st.image(a)


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
        output_text = labels[np.argmax(val[0])]
        font_size = "24px"
        st.markdown("<h4 style='text-align: left; color: #2F3130; font-size: {};'>{}</h4>".format(font_size, output_text), unsafe_allow_html=True)
    except:
        pass
