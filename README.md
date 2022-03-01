## Multi stage filtering for Xray image

During the Covid19 pandemic we have seen many Machine Learning models developed for Covid inference based on Chest X-ray images. We have tried to work with crowd sourced Chest Xray images, i.e pictures of Chest Xray captured from mobile phone cameras. Such images need to go through a filtering stage before feeding it to inference networks. \
While institutionally sourced images are mostly santized and well cropped as they are digitally sourced and can be readily given as input to the inference models, crowd sourced images comes with their own set of challenges such as poor cropping, high tint, presence of human body part such as hands/fingers, skewed geometry etc.. \
Crowd sourced images, thus, need a filtering process before they can be sent to the inference network. In this work we have come up with a multi stage filtering framework for preprocessing crowd sourced chest X-ray images before they are sent to the inference network

Multi stage filtering involves 3 tasks
1. Identify if the image is Xray or not*
2. If the image is an Xray, classify it as good/bad image/uncertain
3. If the Xray is a good image, crop the region of interest, i.e the lung region 
4. You can pass the cropped image to the inference network of your choice \
<small>*stage 1 code will be updated shortly, please use only Chest Xray images for running the scripts detailed below</small>

We trained five different CNN models(Resnet, VGG, Xception, Inception and Mobilenet) for the Stage 2 & 3 and pooled their result by voice vote and Non-Max supression to get the image class and the image bounding box if the image is good. \
Below we describe how to get the results on your dataset.

## Result
1. ### Script
    - Place the Chest Xray images in a folder
    - Run the script **expernet_inference.py** by pointing to the above folder \
    `python expertnet_inference.py -i <path-chest-xray-image-folder>`
    - Results are stored in the **result** folder where the script places the images in one of **good/bad/uncertain** folders. If the image is good, cropped image is placed in the **cropped_image** folder

2. ### Streamlit tool
    - Place the Chest Xray image folder within a parent folder
    - `cd streamlit`
    - `streamlit run streamlit_demo.py -- --i <path-parent-image-folder>`
    - Open the url that is shown in the output above
    - Select an image either from a folder or upload them
    - Results are automatically shown
    - If the image is good, tool also allows you to adjust the crop of the image
    - Image results are saved in the result folder based on their class
    \
    *Note: To use streamlit, images should be in a subfolder of the above folder, you can have as many subfolders as desired* 

    \
    In the streamlit tool, you can either upload an image or choose the image from a folder to check the result. In order to instruct the streamlit to use the source of your choice, clear the previously selected source by choosing the 'None' option in the case of image folders or by clearing the file selected if you have uploaded an image.

    *Note: Some test images sourced from Kaggle are placed in **test_images** folder for readily using the script and the tool*
