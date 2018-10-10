from keras.layers import Input,Conv2D,Conv2DTranspose,SeparableConv2D,AveragePooling2D,MaxPooling2D,concatenate,Dropout
from keras.layers import GlobalAveragePooling2D,Reshape,multiply,UpSampling2D,Lambda,Concatenate
from keras.layers import Permute,Add,BatchNormalization,Activation
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

def spatial_attention(input_feature,name=''):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3,name=name+'spatial_attn_concat')([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False,
                    name=name+'spatial_attn_conv')(concat) 
    assert cbam_feature.shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return cbam_feature


def GAU(inputs_high,inputs_low,name=''):
    height=inputs_low._keras_shape[1]
    width=inputs_low._keras_shape[2]
    depth=inputs_low._keras_shape[3]
    global_pool=GlobalAveragePooling2D()(inputs_high)
    global_pool=Reshape((1,1,inputs_high._keras_shape[-1]))(global_pool)
    conv1x1 = Conv2D(depth, (1, 1),activation='relu', padding='same', use_bias=False)(global_pool)
    conv3x3=Conv2D(depth, (3, 3),padding='same', use_bias=False)(inputs_low)
    mul=multiply([conv3x3,conv1x1])
    spatial_feature=spatial_attention(mul)
    
    inputs_high=Conv2D(depth, (3, 3),padding='same', use_bias=False)(inputs_high)
    #with tf.device('/cpu:0'):
    inputs_high=UpSampling2D((height//inputs_high.shape[1],width//inputs_high.shape[2]),name=name+'up_GAU')(inputs_high)
    sum=Add(name=name+'add_GAU')([inputs_high,spatial_feature])
    return mul

def InstruAttention(classes,shape=(480,640,3)):
	vgg16=VGG16(include_top=False,weights='imagenet',input_shape=shape,classes=1000)
	for layer in vgg16.layers:
		layer.trainable=False
	model=vgg16.output

	model=Conv2D(256, (3, 3), padding='same')(model)
	model=BatchNormalization(epsilon=1e-5)(model)
	model=Activation('relu')(model)
	model = Dropout(0.1)(model)

	model_tr=Conv2DTranspose(256,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([GAU(model,vgg16.layers[17].output,'first'),model_tr],axis=3)

	model=Conv2D(256, (3, 3), padding='same')(model)
	model=BatchNormalization(epsilon=1e-5)(model)
	model=Activation('relu')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(256,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([GAU(model,vgg16.layers[13].output,'second'),model_tr],axis=3)

	model=Conv2D(128, (3, 3), padding='same')(model)
	model=BatchNormalization(epsilon=1e-5)(model)
	model=Activation('relu')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(128,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([GAU(model,vgg16.layers[9].output,'third'),model_tr],axis=3)

	model=Conv2D(64, (3, 3), padding='same')(model)
	model=BatchNormalization(epsilon=1e-5)(model)
	model=Activation('relu')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(64,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([GAU(model,vgg16.layers[5].output,'fourth'),model_tr],axis=3)

	model=Conv2D(32, (3, 3), padding='same')(model)
	model=BatchNormalization(epsilon=1e-5)(model)
	model=Activation('relu')(model)
	model = Dropout(0.1)(model)
	model_tr=Conv2DTranspose(32,(3,3),strides=2,padding='same',activation='relu')(model)
	model=concatenate([GAU(model,vgg16.layers[2].output,'fifth'),model_tr],axis=3)

	model=Conv2D(32, (3, 3), activation='relu', padding='same')(model)

	model=Conv2D(classes, (1, 1), padding='same')(model)
	instru = Model(inputs = vgg16.input, outputs = model)
	return instru

#instru=InstruAttention(4)
#plot_model(instru, to_file='instru_attention.png',show_shapes=True)