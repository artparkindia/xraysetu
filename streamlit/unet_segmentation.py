import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import Model, load_model
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Concatenate
from tensorflow.keras.layers import Dense, AveragePooling2D, Input, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# if gpus:
#    try:
#        tf.config.experimental.set_virtual_device_configuration(gpus[0], 
#                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
#     except RuntimeError as e:
#        print(e)
        
def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Concatenate(axis=3)([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=3)([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=3)([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=3)([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])

def load_images_new(img, size = (256, 256)): 
    img = cv2.resize(img,size)[:,:,0]
    return np.array([img]).reshape((-1,size[0],size[1],1))

def clustered_img(img):
    xarr, yarr = [], []
    for i in range(256):
        for j in range(256):
            if img[i][j] > 0.5:
                xarr.append(i)
                yarr.append(j)
    lung_region = np.vstack([xarr, yarr]).T
    clusters = DBSCAN(eps=1.0,min_samples=3).fit(lung_region)
    initial_result = clusters.labels_
    num_pts = lung_region.shape[0]
    indices = {}
    for i in range(num_pts):
        if int(initial_result[i]) not in indices:
            indices[int(initial_result[i])] = [i]
        else:
            indices[int(initial_result[i])].append(i)
    # Get convex hulls for each cluster
    hulls = {}
    for i in indices:
        pts = np.unique(lung_region[indices[i]], axis=0)
        hull = ConvexHull(pts)
        hulls[i] = hull, pts
    return hulls

def get_avg_hull_score(img_file):
    hull_area_arr = []
    hull_area_file = []
    preds = model.predict(img_file)
    good_images = 0
    bad_images = []
    for i, pred_img in enumerate(preds):
        try:
            lung_area = clustered_img(pred_img)
            hull_area= []
            for j in range(len(lung_area)):
                hull, pts = lung_area[j]
                np_pts = np.array(pts)
                hull_area.append(100*hull.volume/(256**2))           
            hull_area = sorted(hull_area,reverse=True)[:2]
            hull_area_arr.append(hull_area)
            if np.min(hull_area) >= 7.5:
                good_images += 1
            else:
                bad_images.append(i)
        except:
            bad_images.append(i)
    return hull_area_arr, good_images

@st.cache
def get_unet_score(image):
    img_list = load_images_new(image)
    res = get_avg_hull_score(img_list)
    return res

model = unet(input_size=(256, 256,1))
weight_path="../trained_models/cxr_reg_weights.best.hdf5"
model.load_weights(weight_path)
