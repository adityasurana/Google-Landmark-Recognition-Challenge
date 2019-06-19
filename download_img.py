#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import multiprocessing
import os
from io import BytesIO
from urllib import request
import pandas as pd
import re
import tqdm
from PIL import Image
OUT_DIR= 'train'
TARGET_SIZE = 96
  # image resolution to be stored
IMG_QUALITY = 90
def download_image(key_url):
    
    (key, url,lid) = key_url
    lid=str(lid) 
    fold=os.path.join(OUT_DIR,lid) 
    if not os.path.exists(fold):
        os.mkdir(fold)
    
    filename = os.path.join(fold, '{}.jpg'.format(key))
    
    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        pil_image = pil_image.resize((TARGET_SIZE, TARGET_SIZE))
    except:
        print('Warning: Failed to resize image {}'.format(key))
        return 1

    try:
        pil_image.save(filename, format='JPEG', quality=IMG_QUALITY)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1

    return 0

