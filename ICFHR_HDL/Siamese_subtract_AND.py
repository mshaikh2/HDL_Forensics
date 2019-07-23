
# coding: utf-8

# In[1]:

import os
import cv2 as ocr
import pandas as pd
from os import path
import shutil
from tqdm import tqdm,trange
import sys
import gc
import os
from numpy.random import choice
from itertools import combinations
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import math
from tqdm import tqdm
import keras as K
import matplotlib.pyplot as plt
from collections import Counter
from keras.utils import np_utils
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Activation,BatchNormalization,Lambda, Input, Conv2D, Dense,MaxPooling2D, Dropout, Flatten, UpSampling2D, Reshape


def getNextbatch(batch_size=10, stepNumber=0, imgDim = 128, imageFiles = 'list/of/files', pathSplitIndex = 2, classValLength = 4):
    import cv2 as ocr
    try:
        input_label_list = []
        input_data_list = []
        counter = 0
        start = stepNumber
        end = stepNumber + batch_size

        for imgF in imageFiles[start:end]:
          
            input_label_list.append(int(imgF.split('/')[pathSplitIndex][0:classValLength]))
            input_data_list.append(ocr.imread(imgF,0))

        return (np.array(input_data_list).reshape(-1,imgDim,imgDim,1)
                , np.array(input_label_list).astype('int64'))
    except Exception as e:
        print (e)
        
def getData(dataPath='path/To/Data',channels = 0, pathSplitIndex = 2, classValLength = 4, classCount = None, classValue = None):
    from os import listdir
    from os.path import isfile, join
    
    from tqdm import tqdm
    imageClass = []
    imageFiles = [] 
    input_label_list = []
    input_data_list = []   
    selectedFilePaths = []
    imageFiles = [dataPath+'/'+f for f in listdir(dataPath) if isfile(join(dataPath, f))]
    
    for imgF in tqdm(imageFiles, total=len(imageFiles), unit="files"): 
#         print(imgF)
        c = int(imgF.split('/')[pathSplitIndex][0:classValLength])
        if( classCount!=None and classValue == None):
            imageClass.append(c) 
            if(len(Counter(imageClass).keys())>classCount):
                imageClass[classCount] = imageClass[classCount-1]
            else:
#                 input_label_list.append(c)
#                 input_data_list.append(ocr.imread(imgF,channels))
                selectedFilePaths.append(imgF)
        elif(classCount==None and classValue!=None):
            if(c in classValue):
#                 input_label_list.append(c)
#                 input_data_list.append(ocr.imread(imgF,channels))
                selectedFilePaths.append(imgF)
        imageClass = list(set(imageClass))
    return np.array(selectedFilePaths)

def contrastive_loss(left_model, right_model, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(left_model,right_model), 2), 1, keep_dims=True))
        part1= y * tf.square(d)    
        part2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
        return tf.reduce_mean(part1 + part2) /2
    

def gen_random_train_batch(nb_examples):
    out_l,out_r,out_y = [],[],[]
    li = random.sample(range(len(df)), nb_examples)
    for i in li:
        out_l.append(ocr.imread(df.iloc[i]['left'],0))
        out_r.append(ocr.imread(df.iloc[i]['right'],0))
        out_y.append(df.iloc[i]['label'])
    out_l = ((255.01 - np.array(out_l))/255.0).astype('float32')
    out_r = ((255.01 - np.array(out_r))/255.0).astype('float32')
    out_y = np.array(out_y).reshape((-1,1))
    return out_l.reshape(-1,imDim,imDim,1), out_r.reshape(-1,imDim,imDim,1), np.array(out_y)

def gen_random_val_batch(nb_examples):
    out_l,out_r,out_y = [],[],[]
    li = random.sample(range(len(df_val)), nb_examples)
    for i in li:
        out_l.append(ocr.imread(df_val.iloc[i]['left'],0))
        out_r.append(ocr.imread(df_val.iloc[i]['right'],0))
        out_y.append(df_val.iloc[i]['label'])
    out_l = ((255.01 - np.array(out_l))/255.0).astype('float32')
    out_r = ((255.01 - np.array(out_r))/255.0).astype('float32')
    out_y = np.array(out_y).reshape((-1,1))
    return out_l.reshape(-1,imDim,imDim,1), out_r.reshape(-1,imDim,imDim,1), np.array(out_y)

from itertools import combinations
def generateDatasetAndSave_Siamese():
    label = []
    left_path = []
    right_path = []
    simCounter = 0
    for l,r in combinations(train_files_list,2):
        if(int(l.split('/')[3][:4]) == int(r.split('/')[3][:4])):
            label.append([1])
    #     else:
    #         label.append([0,1])
            left_path.append(l)
            right_path.append(r)
            simCounter+=1
    diffCounter = 0
    for l,r in combinations(train_files_list,2):
        if(int(l.split('/')[3][:4]) != int(r.split('/')[3][:4])):
            label.append([0])
    #     else:
    #         label.append([0,1])
            left_path.append(l)
            right_path.append(r)
            diffCounter+=1
        if(diffCounter==simCounter):
            break
    df = pd.DataFrame()
    pd.options.display.max_colwidth = 100
    df['left'] = left_path
    df['right'] = right_path
    df['label'] = label
    df.to_csv('dataset_training_siamese.csv')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
imDim = 64
input_shape  = (imDim,imDim,1)
inp_img = Input(shape = (imDim,imDim,1), name = 'ImageInput')
model = inp_img

#     model = Input(shape=(imDim,imDim,1))
#     model.add(Input(shape = (imDim,imDim,1), name = 'FeatureNet_ImageInput'))
model = Conv2D(32,kernel_size=(3, 3),activation='relu',input_shape=input_shape,padding='valid')(model)
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model = MaxPooling2D((2,2), padding='valid')(model)
model = Conv2D(64, (3, 3), activation='relu',padding='valid')(model)
#     model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
model = MaxPooling2D((2,2),padding='valid')(model)
#     model.add(Conv2D(16, (3, 3), activation='relu',padding='same'))
model = Conv2D(128, (3, 3), activation='relu',padding='valid')(model)
model = MaxPooling2D((2,2),padding='valid')(model)
#     model.add(Conv2D(1, (3, 3), activation='relu',padding='same'))
#     model.add(Conv2D(2, (3, 3), activation='relu',padding='same'))

model = Conv2D(256, (1, 1), activation='relu',padding='valid')(model)
model = MaxPooling2D((2,2),padding='valid')(model)

model = Conv2D(64, (1, 1), activation='relu',padding='valid')(model)
# model = MaxPooling2D((2,2),padding='valid')(model)
model = Flatten()(model)

# img_in = np.array((-1,imDim,imDim,1), dtype='float32')
# img_in = tf.placeholder(shape=(imDim,imDim,1), dtype='float32')

feat = Model(inputs=[inp_img], outputs=[model],name = 'Feat_Model')
feat.summary()


# In[27]:

left_img = Input(shape = (imDim,imDim,1), name = 'left_img')
right_img = Input(shape = (imDim,imDim,1), name = 'right_img')


# In[28]:

left_feats = feat(left_img)
right_feats = feat(right_img)


# In[35]:

from keras.layers import subtract
import random


# In[36]:

merged_feats = subtract([left_feats, right_feats], name = 'subtracted_feats')
merged_feats = Dense(1024, activation = 'linear')(merged_feats)
merged_feats = BatchNormalization()(merged_feats)
merged_feats = Activation('relu')(merged_feats)
merged_feats = Dense(4, activation = 'linear')(merged_feats)
merged_feats = BatchNormalization()(merged_feats)
merged_feats = Activation('relu')(merged_feats)
merged_feats = Dense(1, activation = 'sigmoid')(merged_feats)
similarity_model = Model(inputs = [left_img, right_img], outputs = [merged_feats], name = 'Similarity_Model')
similarity_model.summary()


# In[37]:

sgd = K.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
similarity_model.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics = ['mae'])
similarity_model.load_weights('weights/weights-siamese-AND-subtract-cnn-0.128636.hdf5')

# In[38]:

df = pd.DataFrame.from_csv('dataset_training_siamese.csv')
df_val = pd.DataFrame.from_csv('dataset_validation_siamese.csv')
df.shape[0]


# In[39]:

# iteration = 0
# left_image = ocr.imread(df.iloc[iteration]['left'],0)
# right_image = ocr.imread(df.iloc[iteration]['right'],0)
# y = df.iloc[iteration]['label']


# In[40]:
n_epochs = 500
batch_size = 128
imgDim =  64
best_loss_tr = np.infty
train_files_list = df.shape[0]
train_num_examples = train_files_list
print('Train on: %2d, Validate on: %2d' %(train_num_examples, df_val.shape[0]))
n_iterations_per_epoch = train_num_examples // batch_size
# n_iterations_validation = val_num_examples // batch_size
# Set's how much noise we're adding to the MNIST images
noise_factor = 0.0
sess = tf.Session()
sess.run(tf.global_variables_initializer())
loss_vals = []
acc_vals = []

      

batch_metrics = None
batch_metric_loss = []
batch_metric_acc = []
for epoch in range(n_epochs):
    for iteration in range(int(train_files_list/batch_size)):
        left_image ,right_image, y = gen_random_train_batch(batch_size)
        left_image = left_image.reshape((-1, imgDim, imgDim, 1))
        right_image = right_image.reshape((-1, imgDim, imgDim, 1))
        batch_metrics = similarity_model.train_on_batch([left_image,right_image],[y])
        batch_metric_loss.append(batch_metrics[0])
        batch_metric_acc.append(batch_metrics[1])
        print("\rTraining the model: {}/{} ({:.1f}%)".format(
                  iteration, n_iterations_per_epoch,
                  iteration * 100 / n_iterations_per_epoch),
              end="" * 10)
        
    loss_tr = np.mean(batch_metric_loss)
    acc_tr = np.mean(batch_metric_acc)
    v_batch_size = 500
    v_left_image ,v_right_image, v_y = gen_random_val_batch(100)
    pred_sim = similarity_model.predict([v_left_image ,v_right_image])
    v_acc = (pred_sim.round().astype('int')==v_y).sum()*100.0/float(len(pred_sim))    
    
    print("\rEpoch: {}  Train accuracy: {:.4f}%, Validation acc: {:.4f}%, Loss: {:.6f}{}".format(
        epoch + 1, (1-acc_tr) * 100, v_acc, loss_tr,
        " (improved)" if loss_tr < best_loss_tr else ""))

        
    
    if loss_tr < best_loss_tr and ((epoch % 10)==0):
#             save_path = saver.save(sess, checkpoint_path)
        best_loss_tr = loss_tr
        filepath="weights/weights-siamese-AND-subtract-cnn-"+str(loss_tr)+".hdf5"
        similarity_model.save(filepath)
#     df = pd.DataFrame()
#     df['accuracy'] = batch_metric_acc
#     df['loss'] = batch_metric_loss

