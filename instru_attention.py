from keras.layers import Input,Conv2D,Conv2DTranspose,SeparableConv2D,AveragePooling2D,MaxPooling2D,concatenate,Dropout
from keras.layers import GlobalAveragePooling2D,Reshape,multiply
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import plot_model
from loss import IOU_calc,IOU_calc_loss
from data import dataProcess
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import array_to_img
import numpy as np
from keras.applications.vgg16 import VGG16

def GAU(inputs_high,inputs_low):
    height=inputs_low._keras_shape[1]
    width=inputs_low._keras_shape[2]
    depth=inputs_low._keras_shape[3]
    global_pool=GlobalAveragePooling2D()(inputs_high)
    global_pool=Reshape((1,1,inputs_high._keras_shape[-1]))(global_pool)
    conv1x1 = Conv2D(depth, (1, 1),activation='relu', padding='same', use_bias=False)(global_pool)
    conv3x3=Conv2D(depth, (3, 3),padding='same', use_bias=False)(inputs_low)
    mul=multiply([conv3x3,conv1x1])
    return mul

def InstruAttention(classes,shape=(1024,1280,3)):
	vgg16=VGG16(include_top=False,weights='imagenet',input_shape=shape,classes=1000)
	for layer in vgg16.layers:
		layer.trainable=False
	model=vgg16.output
	b0=Conv2D(256, (1, 1), activation='relu', padding='same')(model)
	b1=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(6))(model)
	b2=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(12))(model)
	b3=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(18))(model)
	b4=AveragePooling2D(pool_size=(2,2),padding='same')(model)
	b4=Conv2D(256,(1,1),activation='relu',padding='same')(b4)
	b4=Conv2DTranspose(256,(3,3),strides=2,padding='same')(b4)
	model=concatenate([b0,b1,b2,b3,b4],axis=3)

	model=Conv2D(256, (3, 3), activation='relu', padding='same')(model)
	model = Dropout(0.1)(model)

	model_tr=Conv2DTranspose(256,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([GAU(model,vgg16.layers[17].output),model_tr],axis=3)

	model=Conv2D(256, (3, 3), activation='relu', padding='same')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(256,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([GAU(model,vgg16.layers[13].output),model_tr],axis=3)

	model=Conv2D(128, (3, 3), activation='relu', padding='same')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(128,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([GAU(model,vgg16.layers[9].output),model_tr],axis=3)

	model=Conv2D(64, (3, 3), activation='relu', padding='same')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(64,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([GAU(model,vgg16.layers[5].output),model_tr],axis=3)

	model=Conv2D(32, (3, 3), activation='relu', padding='same')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(32,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([GAU(model,vgg16.layers[2].output),model_tr],axis=3)

	model=Conv2D(32, (3, 3), activation='relu', padding='same')(model)

	model=Conv2D(classes, (1, 1), padding='same')(model)
	instru = Model(inputs = vgg16.input, outputs = model)
	return instru

#instru=InstruAttention(4)
#plot_model(instru, to_file='instru_attention.png',show_shapes=True)