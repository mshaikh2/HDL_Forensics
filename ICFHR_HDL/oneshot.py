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
import tensorflow as tf
import keras
import pandas as pd
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm, trange
# # imageFiles = TopKimagePathList
# counter = 0
# prevClass = -1
# imageDirTrain = 'Gen2000-Top10-64x64/TrainingSet'
# imageDirVal = 'Gen2000-Top10-64x64/ValidationSet'
# print('Getting Training Data..')
# x_train,y_train = getData(dataPath=imageDirTrain)
# print('Getting Validation Data..')
# x_test,y_test = getData(dataPath=imageDirVal)

# # print(input_label_list)
# nmbr_classes = max(y_train)+1

# y_train = keras.utils.to_categorical(y_train,num_classes=nmbr_classes)
# y_test = keras.utils.to_categorical(y_test,num_classes=nmbr_classes)
# x_train = x_train.astype('float32')
# x_train = (255.1-x_train)/255.0
# x_train = x_train.reshape(-1,64,64,1)
# x_test = x_test.astype('float32')
# x_test = (255.1-x_test)/255.0
# x_test = x_test.reshape(-1,64,64,1)
# x_train,y_train = shuffle(x_train,y_train, random_state=2)



config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

def deepModel(num_classes = 2, input_shape  = (64,64,1)):
    model = Sequential()
    model.add(Conv2D(64,kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128,kernel_size=(3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256,kernel_size=(1, 1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(2,kernel_size=(1, 1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model
# LOG_DIR = 'logs/CNN_VGG'
# tensorCallback = keras.callbacks.TensorBoard(log_dir=LOG_DIR
#                             , batch_size=32
#                             , write_graph=True
#                             , write_grads=True
#                             , write_images=False
#                             , embeddings_freq=0
#                             , embeddings_layer_names=None
#                             , embeddings_metadata=None)


# filepath="weights/inverted-10-Class-valSplit-{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath
#            , monitor='loss'
#            , verbose=1
#            , save_best_only=True
#            , mode='min'
#             ,period = 100
#            , save_weights_only=True)

# callbacks_list = [tensorCallback,checkpoint]
def getData():
    net_dis = deepModel(num_classes=2, input_shape  = (64,64,1))
    net_dis.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
    #               optimizer = keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    net_sim = deepModel(num_classes=2, input_shape  = (64,64,1))
    net_sim.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
    #               optimizer = keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # oneShotModel.summary()

    imgKnown  = ocr.imread(img1,0)
    imgKnown = imgKnown.astype('float32')
    imgKnown = (255.1-imgKnown)/255
    imgKnown = imgKnown.reshape(64,64,1)

    imgQuery  = ocr.imread(img2,0)
    imgQuery = imgQuery.astype('float32')
    imgQuery = (255.1-imgQuery)/255
    imgQuery = imgQuery.reshape(64,64,1)

    return (imgKnown,imgQuery, net_dis, net_sim)

def trainSimandDissim(epochs = 500):
    
    batches = 2
    print('\ntrain dissim start...')
    x = np.array([imgKnown,imgQuery])
    y = [1,0]
    y = keras.utils.to_categorical(y,num_classes=2)
    hist_dis = net_dis.fit(
                           x,
                           y,
                           epochs=epochs,
                           verbose=0,
                           batch_size = batches
                           )


    print('\n================================')
    print('train similar network start...')
    x = np.array([imgKnown,imgQuery])
    y = [1,1]
    y = keras.utils.to_categorical(y,num_classes=2)
    hist_sim = net_sim.fit(
                           x,
                           y,
                           epochs=epochs,
                           verbose=0,
                           batch_size = batches
                           )

def losses():
    print('average losses similar:')
    df = pd.DataFrame(hist_sim.history)
    print( 'acc : %.5f, loss : %.5f ' %(np.average(df['acc']), np.average(df['loss'])))

    print('\naverage losses dissimilar:')
    df = pd.DataFrame(hist_dis.history)
    print( 'acc : %.5f, loss : %.5f ' %(np.average(df['acc']), np.average(df['loss'])))

def similarityScore(printImages = False, seekLayer = 1):
    from keras import backend as K

    inp = net_sim.input   # y = [same class both images]                                        
    outputs = [layer.output for layer in net_sim.layers]          
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  

    imgKnown  = ocr.imread(img1,0)
    imgKnown = imgKnown.astype('float32')
    imgKnown = (255.1-imgKnown)/255
    imgKnown = imgKnown.reshape(64,64,1)

    imgQuery  = ocr.imread(img2,0)
    imgQuery = imgQuery.astype('float32')
    imgQuery = (255.1-imgQuery)/255
    imgQuery = imgQuery.reshape(64,64,1)

    from matplotlib import pyplot as plt
    if (printImages):
        rows = 1
        columns = 2
        fig, axesarr = plt.subplots(rows,columns)
        axesarr[0].imshow(imgKnown.reshape(64,64))
        axesarr[1].imshow(imgQuery.reshape(64,64))
        plt.show()

    test = imgKnown.reshape(-1,64,64,1)
#     print(test.shape)
    layer_outs = [func([test, 1.]) for func in functors]
    im1 = np.array(layer_outs[seekLayer][0])
    # print('features im1 :',im1)

    test = imgQuery.reshape(-1,64,64,1)
#     print(test.shape)
    layer_outs = [func([test, 1.]) for func in functors]
    im2 = np.array(layer_outs[seekLayer][0])
    # print('features im2 :',im2)
    t = 0
    print('\n contrastive loss when y = 1 :sim')
    with tf.Session() as sess:
        t = contrastive_loss(im1,im2,1,margin=0.2).eval()
        print(t)
    return t

    # print('\n contrastive loss when y = 0 :')
    # with tf.Session() as sess:
    #     print(contrastive_loss(im1,im2,0,margin=0.2).eval())

def dissimilarityScore(printImages = False, seekLayer = 1, originShiftBy = 6):
    from keras import backend as K

    inp = net_dis.input     # y = [classifiying in diferent classes]                          
    outputs = [layer.output for layer in net_dis.layers]          
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  

    imgKnown  = ocr.imread(img1,0)
    imgKnown = imgKnown.astype('float32')
    imgKnown = (255.1-imgKnown)/255
    imgKnown = imgKnown.reshape(64,64,1)

    imgQuery  = ocr.imread(img2,0)
    imgQuery = imgQuery.astype('float32')
    imgQuery = (255.1-imgQuery)/255
    imgQuery = imgQuery.reshape(64,64,1)

    from matplotlib import pyplot as plt

    if (printImages):
        rows = 1
        columns = 2
        fig, axesarr = plt.subplots(rows,columns)
        axesarr[0].imshow(imgKnown.reshape(64,64))
        axesarr[1].imshow(imgQuery.reshape(64,64))
        plt.show()

    test = imgKnown.reshape(-1,64,64,1)
#     print(test.shape)
    layer_outs = [func([test, 1.]) for func in functors]
    im1 = np.array(layer_outs[seekLayer][0])
    # print('features im1 :',im1)

    test = imgQuery.reshape(-1,64,64,1)
#     print(test.shape)
    layer_outs = [func([test, 1.]) for func in functors]
    im2 = np.array(layer_outs[seekLayer][0])
    # print('features im2 :',im2)
    t = 0
    print('\n contrastive loss when y = 1 :dissim')
    with tf.Session() as sess:
        t = contrastive_loss(im1,im2,1,margin=0.2).eval()
        print(originShiftBy - t)
    
    return originShiftBy - t

    # print('\n contrastive loss when y = 0 :dissim')
    # with tf.Session() as sess:
    #     print(contrastive_loss(im1,im2,0,margin=0.2).eval())

def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keep_dims=True))
        tmp= y * tf.square(d)    
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
        return tf.reduce_mean(tmp + tmp2) /2

from matplotlib import pyplot as plt
img1 = 'resized_images_handprinted/images/0001a_num1.png'
img2 = 'resized_images_handprinted/images/0002b_num2.png'
imgKnown, imgQuery, net_dis, net_sim = getData()
rows = 1
columns = 2
fig, axesarr = plt.subplots(rows,columns)
axesarr[0].imshow(imgKnown.reshape(64,64))
axesarr[1].imshow(imgQuery.reshape(64,64))
plt.show()
# net_dis.summary()
trainSimandDissim(epochs = 500)
# losses()
# similarityScore(printImages = False, seekLayer=9)
dissimilarityScore(printImages = False, seekLayer=9, originShiftBy =  6)