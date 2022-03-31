import tensorflow as tf
import numpy as np
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_simclr_model():
    saved_model = tf.keras.models.load_model('trained_models/simclr_xray_or_not_aug_tr.tf')
    return saved_model

def xray_or_not(input_image_batch):
    x = tf.cast(input_image_batch, tf.float32)
    model = load_simclr_model()
    res = model.saved_model(x, trainable=False)['final_avg_pool']
    logits_t = model.dense_layer(res)        
    pred = map(bool, tf.argmax(logits_t, axis=1).numpy())
    return pred