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


def Teranus16(classes,shape=(1024,1280,3)):
	vgg16=VGG16(include_top=False,weights='imagenet',input_shape=shape,classes=1000)
	for layer in vgg16.layers:
		layer.trainable=False
	model=vgg16.output
	model=Conv2D(256, (3, 3), activation='relu', padding='same')(model)
	model = Dropout(0.1)(model)

	model_tr=Conv2DTranspose(256,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([vgg16.layers[17].output,model_tr],axis=3)

	model=Conv2D(256, (3, 3), activation='relu', padding='same')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(256,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([vgg16.layers[13].output,model_tr],axis=3)

	model=Conv2D(128, (3, 3), activation='relu', padding='same')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(128,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([vgg16.layers[9].output,model_tr],axis=3)

	model=Conv2D(64, (3, 3), activation='relu', padding='same')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(64,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([vgg16.layers[5].output,model_tr],axis=3)

	model=Conv2D(32, (3, 3), activation='relu', padding='same')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(32,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([vgg16.layers[2].output,model_tr],axis=3)
	model=Conv2D(32, (3, 3), activation='relu', padding='same')(model)
	model=Conv2D(classes, (1, 1), padding='same')(model)
	instru = Model(inputs = vgg16.input, outputs = model)
	return instru

instru=Teranus16(4)
plot_model(instru, to_file='teranus.png',show_shapes=True)