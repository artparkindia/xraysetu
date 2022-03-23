import tensorflow as tf
import numpy as np
import streamlit as st
import streamlit_cropper as st_cropper

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_simclr_model():
    saved_model = tf.keras.models.load_model('../trained_models/simclr_xray_or_not_model.tf')
    return saved_model

def xray_or_not(input_image):
    image = input_image.copy()
    model = load_simclr_model()
    image = tf.image.resize(image,(224, 224))[:,:,:3]
    image = np.expand_dims(image, axis=0)
    image = image/255
    res = model.saved_model(image, trainable=False)['final_avg_pool']
    logits_t = model.dense_layer(res)
    pred = tf.argmax(logits_t, axis=1).numpy()[0]
    return bool(pred)

