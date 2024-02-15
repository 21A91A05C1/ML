# -*- coding: utf-8 -*-
"""
Created on Mon oct 18 11:47:01 2020

@author: Rajesh
"""


import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from time import sleep
##from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import tensorflow as tf
##from tensorflow.keras.models import Sequential 

json_file = open(r'D:\project\modelGG.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

from tensorflow.keras.models import model_from_json
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(r'D:\project\model1GG.h5')
print("Loaded model from disk")


class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.image as plt


###########################################################
test_image=image.load_img(r'D:\project\train\angry\0.jpg',
            target_size=(48,48,1))
###########################################################

test_image = np.array(test_image)
gray = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
test_image=image.img_to_array(gray)
test_image=np.expand_dims(test_image,axis=0)
#print(test_image)
result=loaded_model.predict(test_image)
print(result)

a=list(result[0]).index(max(list(result[0])))
r=class_labels[a]
print(r)

import random
s=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]

if r in class_labels:
    print(random.choice(s[class_labels.index(r)]))



  
