import sys
import os
import numpy as np
import time
import random
import h5py

from keras.models import Model, Sequential,save_model
from keras.layers.normalization import BatchNormalization
from keras.layers import (
    Dense,
    Activation,
    Dropout,
    Conv2D, 
    MaxPooling2D,
    Input,
    Flatten) 
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator                 
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping,TensorBoard, ModelCheckpoint

import tensorflow as tf                                                                                       
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

import resnet
from constants import MODEL_PATH 


def mkdir(path):
    isExist = os.path.exists(path)
    if isExist:
        print("the path exists")
        return False
    else:
        os.makedirs(path)
        print("the path is created successfully")
    return True

def train_init(p_weight = 1):
    mkdir(MODEL_PATH)
    mkdir(MODEL_PATH+'model_store/')
    # input image dimensions
    img_rows, img_cols, img_channels= [4,4,1] 

    # create a resnet network
    model = resnet.ResnetBuilder.build_resnet_4((img_rows, img_cols, img_channels),(4,1))

    # compile and plot the network 
    model.compile(loss = ['categorical_crossentropy' ,'mse'],
            optimizer = 'adam',
            loss_weight = [p_weight, 1])
    plot_model(model, to_file = MODEL_PATH+'model_store/model.png', 
            show_shapes =True,
            show_layer_names = True)
    model.save(MODEL_PATH+'model_store/best_model.h5py')
    model.save(MODEL_PATH+'model_store/old_model.h5py')
    print("The NN model is reset!")

def train_step(data, epochs):
    model = load_model(MODEL_PATH + 'model_store/best_model.h5py')
    os.remove(MODEL_PATH + 'model_store/old_model.h5py')
    os.rename(MODEL_PATH + 'model_store/best_model.h5py', 
            MODEL_PATH + 'model_store/old_model.h5py')

    X_train= data["feature"]
    P_output = data["label"]["P"]
    S_output = data["label"]["S"]

    X_train = np.array(X_train) 
    X_train = np.expand_dims(X_train, axis=3)
    P_output = np.array(P_output)
    S_output = np.array(S_output)
    
    Y_train = [P_output, S_output]
    print(P_output)

    ############################### network parameters ###################################
    batch_size = 32 
    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=1000, min_lr=0.5e-6)
    # early_stopper = EarlyStopping(min_delta= 0.1, patience=200)
    csv_logger = CSVLogger(MODEL_PATH + 'logs.csv')

    # tensorboard = TensorBoard(log_dir = MODEL_PATH+'tensorboard',
            # write_graph = True)
    # checkpoint = ModelCheckpoint(filepath=MODEL_PATH+'model_store/best_model.hdf5',
            # monitor = 'loss',
            # save_best_only = True,
            # save_weights_only = False,
            # mode = 'min',
            # period = 1)
    # callback_list = [lr_reducer, early_stopper,  csv_logger, tensorboard, checkpoint]
    callback_list = [csv_logger]
    #######################################################################################

    train_step = model.fit(X_train, Y_train,
            batch_size =batch_size,
            epochs = epochs,
            shuffle = True,
            callbacks = callback_list)

    # mpe = train_step.history['val_MPE']
    # min_mpe = float('%.2f' % np.min(mpe))
    # print("min_MPE is :", min_mpe)

    model.save(MODEL_PATH+'model_store/best_model.h5py')
    print("training and saving is completed!")


def obtain_model(path=MODEL_PATH + 'model_store/best_model.h5py'):
    model = load_model(path)
    return model

def prediction(matrix, model):
    matrix = np.array(matrix)
    matrix = np.expand_dims(matrix, axis=2)
    matrix = np.expand_dims(matrix, axis=0)
    P,S = model.predict(matrix)
    return P, S
