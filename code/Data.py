import os
import re
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

groups_folder_path = './sample_data/'
categories = ["bed", "cabinet", "chair", "door",
              "refrigerator", "Sofa", "table", "tv", "window"]

num_classes = len(categories)

image_w = 28
image_h = 28

X = []
Y = []

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path + categorie + '/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            img = cv2.imread(image_dir+filename)
            print(image_dir+filename)
            img = cv2.resize(img, (image_w, image_h))
            X.append(img / 255.0)
            Y.append(label)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

# 데이터 형태 변환
# X_train = np.array(X_train)
# X_test = np.array(X_test)
# Y_train = np.array(Y_train).reshape(-1)  # 1차원 배열로 변환
# Y_test = np.array(Y_test).reshape(-1)    # 1차원 배열로 변환

np.save("./x_train_data.npy", X_train)
np.save("./x_test_data.npy", X_test)
np.save("./y_train_data.npy", Y_train)
np.save("./y_test_data.npy", Y_test)
