from tensorflow.keras.utils import Sequence
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import cv2

IMG_WIDTH  = 256
IMG_HEIGHT = 256
base_image_path = "/home/" 

class DataGenerator(Sequence):
    def __init__(self, 
                 filename_list, 
                 bbox_coord=[], 
                 batch_size=32, 
                 path=base_image_path, 
                 input_shape=(256,256,3), 
                 shuffle=True,
                 test_set=False
                ):
        self.batch_size = batch_size
        self.y = bbox_coord
        self.x = np.array(filename_list)
        self.shuffle = shuffle
        self.indexes = np.arange(self.x.shape[0])
        self.path = path
        self.on_epoch_end()
        self.img_h, self.img_w, _ = input_shape
        self.test_set = test_set
    

    def __len__(self):
        return int(np.ceil(self.x.shape[0]/self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:            
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, idx):
        index_list = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.x[index_list]
        batch_X = self.get_image_data(batch_x)
        if not self.test_set:
            current_batch_size = len(batch_x)
            classification=np.empty((current_batch_size), dtype=np.float32)
            bbox = np.empty((current_batch_size, 4), dtype=np.float32)
            for idx, image_name in enumerate(batch_x):
                image_details = self.y[image_name]
                classification[idx] = image_details[0]
                bbox[idx] = image_details[1:]
            batch_y = classification, bbox
            return batch_X, batch_y
        else:
            return batch_X
    
    def get_image_data(self,batch_x):
        
        current_batch_size = len(batch_x)
        X = np.empty((current_batch_size, self.img_h, self.img_w, 3))
        for i, filename in enumerate(batch_x):
            path = os.path.join(self.path, filename)
            if os.path.exists(path):
                image = cv2.imread(path)
                image = cv2.resize(image, (self.img_h,self.img_w))
                image = image/255            
                if(len(image.shape)<3):continue
                X[i,] = image
            else:print(f'{path} does not exist')
        return X