import streamlit as st
import streamlit_cropper as st_cropper
import cv2
import matplotlib.pyplot as plt
import simclr_xray_type_check
import multinet_result
import unet_segmentation
import numpy as np
import time
import json
import os
import io
import sys
import tensorflow as tf
import shutil
import argparse

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

voice_vote_interpretation={
    0:'Bad',
    1:'Good',
    2:'Uncertain'
}

saved_model_name = ["resnet", "vgg", "inception", "xception", "mobilenet"]
result_cropped = 'cropped_image'
trained_models_path = '../trained_models'
test_image_path = '../test_images'

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-image-folder',
                    required=False,
                    type=str,
                    dest="image_path",
                    default=test_image_path,
                    metavar="image_path",
                    help="Specify the image folder path")

args = parser.parse_args()
image_path = args.image_path

result_path = {
    0:'result/bad/',
    1:'result/good/',
    2:'result/uncertain/'
}
cropped_path = 'result/cropped_image/'
#Limit GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], 
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
    except RuntimeError as e:
        print(e)
        
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_trained_model():
    model_={}
    for model_name in saved_model_name:
        st.write(f'loading {model_name}')
        path = os.path.join(trained_models_path, model_name+'.tf')
        tmp_model = tf.keras.models.load_model(path)
        model_[model_name] = tmp_model
    st.write('all models are loaded, upload/select the image')
    return model_

def box_algorithm(coord, w, h):
    def get_bbox_coord(img=None, aspect_ratio=None, **kwargs):
        bbox_dict = {}
        startX, startY, endX, endY = multinet_result.get_bbox_upscaled(coord, w, h)
        bbox_dict['left'] = startX
        bbox_dict['top'] = startY
        bbox_dict['width'] = endX-startX
        bbox_dict['height'] = endY-startY
        return bbox_dict
    return get_bbox_coord

@st.cache
def xray_or_not(input_image):
    with st.spinner("Checking if the Image is a Chest X-Ray"):
        res = simclr_xray_type_check.xray_or_not(input_image)
        return res

def get_voice_vote(input_image):
    all_preds_class = np.empty((1, len(saved_model_name)))
    all_preds_bbox = np.empty((1, len(saved_model_name), 4))
    for idx, model_name in enumerate(saved_model_name):
        img = input_image 
        if model_name=='mobilenet':resize=(224, 224)
        else:resize=(256, 256)
        img = cv2.resize(img,resize)
        img = img/255
        img = np.expand_dims(img, axis=0)
        res = model[model_name].predict(img)
        all_preds_class[:, idx] = res[0].squeeze()
        all_preds_bbox[:, idx] = res[1]
    return all_preds_class, all_preds_bbox

with st.spinner('Loading models...'):
    model=load_trained_model()

# Select image by uploading or selecting from a folder
# Upload Image  
upload_image = True      
file_selected = st.sidebar.file_uploader('Upload image')  

st.sidebar.header('OR')

# Select image from a folder
folder_list = os.listdir(image_path)
selected_folder = None
selected_folder = st.sidebar.selectbox('Select a folder', folder_list)
if selected_folder:
    file_list = os.listdir(os.path.join(image_path, selected_folder))
    file_list.insert(0, None) # default option
    image_name = st.sidebar.selectbox('Select a file', file_list)
    if(image_name):
        upload_image = False
        selected_image_path = os.path.join(image_path, selected_folder, image_name)
        input_image = cv2.imread(selected_image_path)
        file_selected = input_image 

realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)

is_xray = None
voice_vote = None
new_image = False

if (not upload_image and file_selected.any()) or (file_selected):
    curr_img_name = file_selected.name if upload_image else image_name
    new_image = ('prev_image' not in st.session_state) or (st.session_state['prev_image']!=curr_img_name)
    if new_image:
        try:
            del st.session_state['class']
            del st.session_state['bbox']
        except Exception as e:
            print('class and bounding box not set')
                       
    st.write('Input image', curr_img_name)
    st.image(file_selected, width=256)
    input_image = np.array(Image.open(file_selected)) if upload_image else file_selected
    is_xray = xray_or_not(input_image)
    'Image is a Chest Xray:', is_xray
    
    st.session_state['prev_image']=curr_img_name    

if is_xray:
    with st.spinner("Checking the Image class(Good/Bad/Uncertain)"):
        try:
            if new_image:
                input_image = input_image[:,:,:3]
                all_preds_class, all_preds_bbox = get_voice_vote(input_image)
                st.session_state['class']=all_preds_class
                st.session_state['bbox']=all_preds_bbox
            else:
                all_preds_class = st.session_state['class']
                all_preds_bbox = st.session_state['bbox']
            voice_vote, bbox_nms_pred = multinet_result.get_voice_vote_nms(all_preds_class, all_preds_bbox)
            st.write(f'Image class:{voice_vote_interpretation[voice_vote].upper()}')
            img = input_image
            plt.imsave(os.path.join(result_path[voice_vote], st.session_state['prev_image']), img)
        except Exception as e:
            st.write('error while processing')
            sys.exit(1)
else:
    voice_vote = -1
    bbox_nms_pred = ''

if voice_vote_interpretation.get(voice_vote, 'NA').upper()=='GOOD':
    with st.spinner("Checking the cropping quality..."):
        try:
            img_ = input_image
            img_ = cv2.resize(img_,(256, 256))
            pil_img = Image.fromarray(img_)
            h, w = img_.shape[:2]
        
            bbox_coord = box_algorithm(bbox_nms_pred, w, h)
            startX, startY, width, height = bbox_coord().values()
            cropped_img = img_[startY:startY+height, startX:startX+width]
            cropped_img = cv2.resize(cropped_img, (256, 256))
            
            hull_area_arr, good_crop = unet_segmentation.get_unet_score(cropped_img)
        except Exception as e:
            st.write('Error in getting unet score')
            sys.exit(1)            
    
    st.write(f'Image crop quality check:{"PASS" if good_crop else "FAIL"}')
             
    if good_crop: 
        col1, col2 = st.columns(2)
        with col1:
            cropped_pred = st_cropper.st_cropper(pil_img, 
                                                 realtime_update=realtime_update, 
                                                 box_color="#00FF00", 
                                                 aspect_ratio=(1,1), 
                                                 box_algorithm=bbox_coord
                                                )
        with col2:  
            st.write('Cropped Image')
#             cropped_pred.thumbnail((256,256))
            st.image(cropped_pred)
            cropped_pred.save(os.path.join(cropped_path, st.session_state['prev_image']))
           
    else:
        st.write('Predicted cropped image')
        st.image(cropped_img)
        st.write('Predicted crop quality below threshold, sending it to the technician for a review')
