from keras.layers import Input,Conv2D,Conv2DTranspose,SeparableConv2D,AveragePooling2D,MaxPooling2D,concatenate,Dense,Flatten
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras import backend as K
from keras.utils import plot_model
from loss import IOU_calc,IOU_calc_loss
from data import dataProcess
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import array_to_img
import numpy as np
from data_tiny_imagenet import DataGenerator

inputs=Input(shape=(64,64,3))
model=Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
model=Conv2D(64, (3, 3), activation='relu', padding='same')(model)
model=MaxPooling2D(pool_size=(2,2))(model)

model=Conv2D(128, (3, 3), activation='relu', padding='same')(model)	
model=Conv2D(128, (3, 3), activation='relu', padding='same')(model)	
model=MaxPooling2D(pool_size=(2,2))(model)

model=Conv2D(256, (3, 3), activation='relu', padding='same')(model)	
model=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(3))(model)
model=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(3))(model)
model=MaxPooling2D(pool_size=(2,2))(model)

model_conn1=Conv2D(512, (3, 3), activation='relu', padding='same')(model)	
model=SeparableConv2D(512, (3,3), activation='relu', padding='same',dilation_rate=(12))(model)
model=SeparableConv2D(512, (3,3), activation='relu', padding='same',dilation_rate=(12))(model)
model=MaxPooling2D(pool_size=(2,2))(model)

b0=Conv2D(512, (1, 1), activation='relu', padding='same')(model)
b1=SeparableConv2D(512, (3,3), activation='relu', padding='same',dilation_rate=(6))(model)
b2=SeparableConv2D(512, (3,3), activation='relu', padding='same',dilation_rate=(12))(model)
b3=SeparableConv2D(512, (3,3), activation='relu', padding='same',dilation_rate=(18))(model)
b4=AveragePooling2D(pool_size=(2,2),padding='same')(model)
b4=Conv2D(512,(1,1),activation='relu',padding='same')(b4)
b4=Conv2DTranspose(512,(3,3),strides=2,padding='same')(b4)
model=concatenate([b0,b1,b2,b3,b4],axis=3)

model=Flatten()(model)
model=Dense(4096)(model)
model=Dense(4096)(model)
model=Dense(200,activation='softmax')(model)



instru = Model(input = inputs, output = model)
instru.compile(optimizer = SGD(lr = 0.1), loss = 'categorical_crossentropy', metrics = ['acc'])


labels_train_dict={}
list_IDs_train = [line.rstrip('\n') for line in open('list_IDs_train.txt')]
labels_train = [line.rstrip('\n') for line in open('labels_train.txt')]
for i in range(len(list_IDs_train)):
	labels_train_dict[list_IDs_train[i]]=labels_train[i]

labels_val_dict={}
list_IDs_val = [line.rstrip('\n') for line in open('list_IDs_val.txt')]
labels_val = [line.rstrip('\n') for line in open('labels_val.txt')]
for i in range(len(list_IDs_val)):
	labels_val_dict[list_IDs_val[i]]=labels_val[i]

params = {'height': 64,
		  'width': 64,
          'batch_size': 256,
          'n_classes': 200,
          'n_channels': 3,
          'shuffle': True}
training_generator = DataGenerator(list_IDs_train, labels_train_dict, **params)
validation_generator = DataGenerator(list_IDs_val, labels_val_dict, **params)

model_checkpoint = ModelCheckpoint('drive/checkpoints/instru_enc.hdf5', monitor='loss',verbose=1, save_best_only=True)
#instru.load_weights('drive/checkpoints/instru_enc.hdf5')
#instru.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=3, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
instru.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    callbacks=[model_checkpoint],
                    epochs=10)

