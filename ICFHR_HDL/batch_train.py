import cv2 as ocr
import sys
import gc
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import math
from tqdm import tqdm
import keras
from keras.utils import np_utils
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense,MaxPooling2D, Dropout, Flatten


imageDir = 'resized_images_handprinted/images'
imageFiles = [imageDir+'/'+f for f in listdir(imageDir) if isfile(join(imageDir, f))]
imageClass = []

filesCount = len(imageFiles)


for imgF in imageFiles:
    imageClass.append(int(imgF.split('/')[2][0:4]))
    
num_classes = max(imageClass)

datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      rescale=1./255,
      fill_mode='nearest')

def getImages(num_images = 10, slideBy = 1, slideCounter = 0):
    try:
        input_label_list = []
        input_data_list = []
        counter = 0
        if (slideCounter+num_images) == len(imageFiles):
            slideCounter = 0
            
        for imgF in imageFiles[slideCounter:slideCounter+num_images]:
            input_label_list.append(int(imgF.split('/')[2][0:4]))
            input_data_list.append(ocr.imread(imgF))
            counter += 1
            if counter == num_images :
                break 
        slideCounter = slideCounter + slideBy
        return np.array(input_data_list), np.array(input_label_list)
    except Exception as e:
        print(e)
    

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
input_shape  = (500,500,3)
model = Sequential()
model.add(Conv2D(32,kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()


df = pd.DataFrame()
batches = 0
b_s = 1
num_samples = 50
epochs = 10
batches = 0
slideBySteps = num_samples - 1
batch_metrics = None
batch_metric_loss = []
batch_metric_acc = []
df = pd.DataFrame()
for epoch in range(1, epochs+1):
    for slideCnt in tqdm(range(0,len(imageFiles)), total=len(imageFiles), unit="files"):
        x_train, y_train = getImages(num_images=num_samples, slideBy=slideBySteps, slideCounter = slideCnt)
        y_train = np_utils.to_categorical(y_train,num_classes)
        for x_batch, y_batch in datagen.flow(x_train,y_train, batch_size=b_s):
            batch_metrics = model.train_on_batch(x_batch, y_batch)
            batches += 1
            if batches >= (num_samples/b_s):
                batches = 0
                break
        batch_metric_loss.append(batch_metrics[0])
        batch_metric_acc.append(batch_metrics[1])
    filepath="weights-deep-cnn.hdf5"
    model.save(filepath)
    df = pd.DataFrame()
    df['accuracy'] = batch_metric_acc
    df['loss'] = batch_metric_loss
    ax = df.plot(subplots=True)
    accChart = ax[0]
    lossChart = ax[1]
    figAcc =  accChart.get_figure()
    figAcc.savefig('AccChart.png')
    figLoss =  lossChart.get_figure()
    figLoss.savefig('LossChart.png')
df.to_csv('loss_acc_metrics.csv')

    