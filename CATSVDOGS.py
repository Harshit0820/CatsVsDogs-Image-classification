import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

import os
print(os.listdir("C:/Users/91999/Desktop/Course work/IMG processing/input"))

import zipfile

with zipfile.ZipFile("C:/Users/91999/Desktop/Course work/IMG processing/input/train.zip","r") as z:
    z.extractall(".")

with zipfile.ZipFile("C:/Users/91999/Desktop/Course work/IMG processing/input/test1.zip","r") as z:
    z.extractall(".")

path = "C:/Users/91999/Desktop/Course work/IMG processing/train/"

for p in os.listdir(path):
    category = p.split(".")[0]
    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array, dsize=(80, 80))
    plt.imshow(new_img_array,cmap="gray")
    break

X = []
y = []
convert = lambda category : int(category == 'dog')
def create_test_data(path):
    for p in os.listdir(path):
        category = p.split(".")[0]
        category = convert(category)
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append(category)

create_test_data(path)
X = np.array(X).reshape(-1, 80,80,1)
y = np.array(y)

X = X/255.0

model = Sequential()
model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=1, batch_size=32, validation_split=0.2)

path = "C:/Users/91999/Desktop/Course work/IMG processing/test1/"

X_test = []
id_line = []
def create_test1_data(path):
    for p in os.listdir(path):
        id_line.append(p.split(".")[0])
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X_test.append(new_img_array)
create_test1_data(path)
X_test = np.array(X_test).reshape(-1,80,80,1)
X_test = X_test/255

predictions = model.predict(X_test)

predicted_val = [int(round(p[0])) for p in predictions]

output_df = pd.DataFrame({'id':id_line, 'label':predicted_val})

output_df.to_csv("output.csv", index=False)

opdf = pd.read_csv('output.csv')
print(opdf)
