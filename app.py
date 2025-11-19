import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import traceback

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("my_model.keras")
    return model

model = load_model()

st.title("Classificador e Analisador de Imagens")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem enviada", use_column_width=True)

    img = img.resize((225, 225)) 
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 

    pred = model.predict(img_array)

    classe_pred = np.argmax(pred, axis=1)
    st.write("Probabilidades:", pred)
    st.write("Classe prevista:", classe_pred)
