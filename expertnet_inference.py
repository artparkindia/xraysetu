import unet_segmentation
import multinet_result
import argparse
import os
import sys
import DataGenerator
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import simclr_xray_type_check
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
saved_model_name = ["vgg", "resnet", "inception", "xception", "mobilenet"]
model_path = 'trained_models'

result_crop = 'result/cropped_image/'

result_path = {
    0:'result/bad/',
    1:'result/good/',
    2:'result/uncertain/',
    3:'result/non_xray/'
}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-image',
                    required=True,
                    type=str,
                    dest="image_path",
                    metavar="image_path",
                    help="Specify the image folder path")

args = parser.parse_args()
image_path = args.image_path

if(not os.path.exists(image_path)):
    print('Image folder path does not exist')
    sys.exit()
    
image_list = os.listdir(image_path)[:300]
if(len(image_list)==0):
    print('Folder is empty')
    sys.exit()

xray_image_idx = []
print('Num of images:', len(image_list))
xray_or_not_test_set = DataGenerator.DataGenerator(image_list, path=image_path, shuffle=False, test_set=True, batch_size=128)
print('Checking Xray or Not...')
for batch in xray_or_not_test_set:
    res = simclr_xray_type_check.xray_or_not(batch)
    xray_image_idx.extend(res)

image_list = np.asarray(image_list)
xray_image_idx = np. asarray(xray_image_idx)

xray_image_list = image_list[xray_image_idx]
non_xray_list = image_list[~xray_image_idx]

for file_name in non_xray_list:
    non_xray_image_path = os.path.join(image_path, file_name)
    shutil.copy(non_xray_image_path, result_path[3])
    
        
if len(xray_image_list)==0:
    print("No chest xray-image found, exiting...")
    sys.exit()

print("Num of chest xray images:", len(xray_image_list)) 
test_set_regular = DataGenerator.DataGenerator(xray_image_list, path=image_path, shuffle=False, test_set=True)
test_set_mobilenet = DataGenerator.DataGenerator(xray_image_list, path=image_path, shuffle=False, test_set=True, input_shape=(224, 224, 3))

#Limit GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], 
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
    except RuntimeError as e:
        print(e)

def load_trained_model():
    model_={}
    for model_name in saved_model_name:
        st.write(f'loading {model_name}')
        path = os.path.join(model_path, model_name+'.tf')
        tmp_model = tf.keras.models.load_model(path)
        model_[model_name] = tmp_model
    return model_


def get_bbox_coord(coord, w, h):
    bbox_dict = {}
    startX, startY, endX, endY = multinet_result.get_bbox_upscaled(coord, w, h)
    bbox_dict['left'] = startX
    bbox_dict['top'] = startY
    bbox_dict['width'] = endX-startX
    bbox_dict['height'] = endY-startY
    return bbox_dict


def get_pred_result():
    all_preds_class = np.empty((len(xray_image_list), len(saved_model_name)))
    all_preds_bbox = np.empty((len(xray_image_list), len(saved_model_name), 4))
    for idx, model_name in enumerate(saved_model_name):
        path = os.path.join(model_path, model_name+'.tf')
        print(f'Loading and running inference on {model_name}')
        if os.path.exists(path):            
            test_data_gen = test_set_mobilenet if model_name=='mobilenet' else test_set_regular
            model = load_model(path)
            res = model.predict(test_data_gen)
            all_preds_class[:, idx] = res[0].squeeze()
            all_preds_bbox[:, idx] = res[1]
        else:
            print(f'{model_name} does not exist')
    return all_preds_class, all_preds_bbox

all_preds_class, all_preds_bbox = get_pred_result()
voice_vote, bbox_nms_pred = multinet_result.get_voice_vote_nms(all_preds_class, all_preds_bbox)

for img_num, res, bbox in zip(xray_image_list, voice_vote, bbox_nms_pred):    
    src = os.path.join(image_path, img_num)
    if res==1:
        image = cv2.imread(src)
        h, w = image.shape[:2]        
        startX, startY, endX, endY = multinet_result.get_bbox_upscaled(bbox, w, h)
        cropped_image = image[startY:endY, startX:endX]
        hull_area_arr, good_crop = unet_segmentation.get_unet_score(cropped_image)
        if not good_crop:res=2
        else:plt.imsave(os.path.join(result_crop, img_num), cropped_image)   
    shutil.copy(src, result_path[res])

# print(all_preds_class, all_preds_bbox)
