import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.python.keras.models import Model, load_model

from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
import tqdm
import cv2
from tensorflow.python.keras.callbacks import Callback
from keras.callbacks import History
from keras import backend as K

img_height, img_width=128,128
nclasses=8
nfilters=64
size_stride=1
size_kernel=2
dilation=4
epochs_value=50


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss

historial = History()

p_unet = RedNet(img_height, img_width,nclasses,nfilters,size_stride,size_kernel,dilation)
p_unet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[dice_coeff, 'accuracy'])
tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(mode='max', filepath='models-dr/pdilated.h5', monitor='acc', save_best_only='True', save_weights_only='True', verbose=1)





fold_accs = []
for train_idx, val_idx in kfold.split(dataset_img):
  print('...nuevo fold...')
  X_img_train_fold= dataset_img[train_idx]
  X_depth_train_fold = depth_result[train_idx]
  y_train_fold = X_mask[train_idx]

  X_img_val_fold = dataset_img[val_idx]
  X_depth_val_fold = depth_result[val_idx]
  y_val_fold = X_mask[val_idx]
  #early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='max')
  p_unet = DilatedNet(img_height, img_width,nclasses,nfilters,size_stride,size_kernel,dilation)
  #compilar el modelo
  p_unet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[dice_coeff, 'accuracy'])
  # Entrenar el modelo
  p_unet.fit([X_img_train_fold,X_depth_train_fold],y_train_fold,validation_data=([X_img_val_fold,X_depth_val_fold],y_val_fold), epochs=epochs_value, callbacks=[historial])
