import os
import boto3
import cv2
import streamlit as st

client = boto3.client('rekognition','ap-south-1')
threshold = 50

def check_xray_label(label_list):
    x_ray = False
    for label in label_list:
        if label['Name']=='X-Ray' and label['Confidence'] > threshold:
            x_ray=True
            break
    return x_ray

def check_xray_or_not(img_bytes):
    is_xray = False
    if type(img_bytes) == bytes:
        result = client.detect_labels(Image={'Bytes':img_bytes})
        if(len(result['Labels'])>0):
            is_xray = check_xray_label(result['Labels'])
    else:
        st.error("Invalid image type, expecting bytes")
    return is_xray
    