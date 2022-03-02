import json
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

with open('/home/ubuntu/preprocessing/case-images/image_paths.json', 'r') as f:
    image_path = json.load(f)

saved_models = ["resnet", "vgg", "inception", "xception", "mobilenet"]
all_preds_class = []
all_preds_bbox = []

def model_predict(img):
    for idx, model_name in enumerate(['mobilenet']):
        path = os.path.join(image_path['saved_model_path'], model_name+'.tf')
        if os.path.exists(path):
            print(f'{model_name}...')
            if model_name=='mobilenet':
                img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            print(img.shape)
            model = load_model(path)
            res = model.predict(img)
            all_preds_class.append(res[0].squeeze())
            all_preds_bbox.append(res[1])
        else:
            print(f'{model_name} does not exist')
    return all_preds_class, all_preds_bbox