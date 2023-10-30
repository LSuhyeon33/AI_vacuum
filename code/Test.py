import os
import re
import glob
import cv2
import numpy as np
import shutil
from numpy import argmax
from keras.models import load_model

categories = ["bed", "cabinet", "chair", "door",
              "refrigerator", "Sofa", "table", "tv", "window"]


def Dataization(img_path):
    image_w = 28
    image_h = 28
    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_w, image_h))
    return (img/255.0)


src = []
name = []
test = []
image_dir = "./test_data/"
for file in os.listdir(image_dir):
    if (file.find('.jpg') != -1):
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))


test = np.array(test)
model = load_model('my_model.keras')

predict_probs = model.predict(test)
predict = np.argmax(predict_probs, axis=1)


for i in range(len(test)):
    print(name[i] + " : , Predict : " + str(categories[predict[i]]))
