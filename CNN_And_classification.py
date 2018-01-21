import cv2 as ocr
import sys
import gc
import numpy as np
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense,MaxPooling2D, Dropout, Flatten
import pandas as pd
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm, trange
imageDir = 'resized_images_handprinted/images'
imageFiles = [imageDir+'/'+f for f in listdir(imageDir) if isfile(join(imageDir, f))]
input_label_list = []
input_data_list = []
try:
    counter = 0
    for imgF in tqdm(imageFiles, total=1500, unit="files"):
        input_label_list.append(int(imgF.split('/')[2][0:4]))
        input_data_list.append(np.array(ocr.imread(imageFiles[0]),dtype='float'))
        gc.collect()
        counter += 1
        if counter == 1500 :
            break
except Exception as e:
    print(e)

# num_classes = len(set(input_label_list))
INP_IMGS = np.array(input_data_list)
LABELS = np.array(input_label_list)
Y = keras.utils.to_categorical(LABELS)
# INP_IMGS = INP_IMGS.astype('float32')
INP_IMGS /= 255
x,y = shuffle(INP_IMGS,Y, random_state=2)


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=2)


num_classes = len(set(input_label_list))
batch_size  = 512
epochs = 10
input_shape  = (500,500,3)
model = Sequential()
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,
                 kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2
#                  ,
#           validation_data=(x_test, y_test)#,
#          callbacks=callbacks_list
          )

df = pd.Dataframe(hist)