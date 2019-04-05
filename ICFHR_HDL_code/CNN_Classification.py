import cv2 as ocr
import sys
import gc
import numpy as np
from os import listdir
from os.path import isfile, join
import keras
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm, trange
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import math
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
input_label_list = []
input_data_list = []
try:
    counter = 0
    for imgF in tqdm(imageFiles, total=len(imageFiles), unit="files"):
        input_label_list.append(int(imgF.split('/')[2][0:4]))
        img = ocr.imread(imageFiles[0],ocr.COLOR_BGR2GRAY)        

        input_data_list.append(ocr.imread(imageFiles[0],ocr.COLOR_BGR2GRAY))
        
#         gc.collect()
#         counter += 1
#         if counter == 1000 :
#             break
except Exception as e:
    print(e)

num_classes = max(input_label_list)+1
INP_IMGS = np.array(input_data_list)
LABELS = np.array(input_label_list)
Y = keras.utils.to_categorical(LABELS)
INP_IMGS = INP_IMGS.astype('float32')
INP_IMGS /= 255
x,y = shuffle(INP_IMGS,Y, random_state=2)


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# x_train = x_train.reshape(-1,x_train.shape[1],x_train.shape[2],1)
# x_test = x_test.reshape(-1,x_test.shape[1],x_test.shape[2],1)



LOG_DIR = 'logs/CNN_32_32_64_64_128_128_D128'
tensorCallback = keras.callbacks.TensorBoard(log_dir=LOG_DIR
                            , batch_size=32
                            , write_graph=True
                            , write_grads=True
                            , write_images=False
                            , embeddings_freq=0
                            , embeddings_layer_names=None
                            , embeddings_metadata=None)


filepath="weights/weights-cnn-{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath
           , monitor='loss'
           , verbose=1
           , save_best_only=True
           , mode='min'
            ,period = 100
           , save_weights_only=True)
callbacks_list = [tensorCallback,checkpoint]

epochs = 1000
batches = 1


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
input_shape  = (64,64,3)
model = Sequential()
model.add(Conv2D(64,kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
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

hist = model.fit(x_train, y_train,
          batch_size=batches,
          epochs=epochs,
          verbose=1,
#                  ,
#            validation_data=(x_test, y_test)#,
         callbacks=callbacks_list
          )
		  
df = pd.DataFrame(hist.history)
ax = df.plot(subplots=True)
accChart = ax[0]
lossChart = ax[1]
figAcc =  accChart.get_figure()
figAcc.savefig('AllData_AccChart.png')
figLoss =  lossChart.get_figure()
figLoss.savefig('AllData_LossChart.png')
df.to_csv('allData_loss_acc_metrics.csv')