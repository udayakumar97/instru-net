from keras.layers import Input,Conv2D,Conv2DTranspose,SeparableConv2D,AveragePooling2D,MaxPooling2D,concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import plot_model
from loss import IOU_calc,IOU_calc_loss
from data import dataProcess
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import array_to_img
import numpy as np

inputs=Input(shape=(1024,1280,3))
model=Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
model_conn5=Conv2D(64, (3, 3), activation='relu', padding='same')(model)
model=MaxPooling2D(pool_size=(2,2))(model_conn5)

model=Conv2D(128, (3, 3), activation='relu', padding='same')(model)	
model_conn4=Conv2D(128, (3, 3), activation='relu', padding='same')(model)	
model=MaxPooling2D(pool_size=(2,2))(model_conn4)

model_conn3=Conv2D(256, (3, 3), activation='relu', padding='same')(model)	
model=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(3))(model_conn3)
model=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(3))(model)
model=MaxPooling2D(pool_size=(2,2))(model)


model_conn2=Conv2D(256, (3, 3), activation='relu', padding='same')(model)	
model=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(6))(model_conn2)
model=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(6))(model)
model=MaxPooling2D(pool_size=(2,2))(model)

model_conn1=Conv2D(256, (3, 3), activation='relu', padding='same')(model)	
model=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(12))(model_conn1)
model=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(12))(model)
model=MaxPooling2D(pool_size=(2,2))(model)

b0=Conv2D(256, (1, 1), activation='relu', padding='same')(model)
b1=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(6))(model)
b2=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(12))(model)
b3=SeparableConv2D(256, (3,3), activation='relu', padding='same',dilation_rate=(18))(model)
b4=AveragePooling2D(pool_size=(2,2),padding='same')(model)
b4=Conv2D(256,(1,1),activation='relu',padding='same')(b4)
b4=Conv2DTranspose(256,(3,3),strides=2,padding='same')(b4)
model=concatenate([b0,b1,b2,b3,b4],axis=3)

model=Conv2D(256, (3, 3), activation='relu', padding='same')(model)
model_tr=Conv2DTranspose(256,(3,3),strides=2,padding='same',activation='relu')(model)
model=concatenate([model_conn1,model_tr],axis=3)

model=Conv2D(256, (3, 3), activation='relu', padding='same')(model)
model_tr=Conv2DTranspose(256,(3,3),strides=2,padding='same',activation='relu')(model)
model=concatenate([model_conn2,model_tr],axis=3)

model=Conv2D(128, (3, 3), activation='relu', padding='same')(model)
model_tr=Conv2DTranspose(128,(3,3),strides=2,padding='same',activation='relu')(model)
model=concatenate([model_conn3,model_tr],axis=3)

model=Conv2D(64, (3, 3), activation='relu', padding='same')(model)
model_tr=Conv2DTranspose(64,(3,3),strides=2,padding='same',activation='relu')(model)
model=concatenate([model_conn4,model_tr],axis=3)

model=Conv2D(32, (3, 3), activation='relu', padding='same')(model)
model_tr=Conv2DTranspose(32,(3,3),strides=2,padding='same',activation='relu')(model)
model=concatenate([model_conn5,model_tr],axis=3)

model=Conv2D(32, (3, 3), activation='relu', padding='same')(model)
model=Conv2D(1, (1, 1), activation='sigmoid', padding='same')(model)


instru = Model(input = inputs, output = model)
instru.compile(optimizer = Adam(lr = 1e-4), loss = IOU_calc_loss, metrics = [IOU_calc])
mydata = dataProcess(1024, 1280)
imgs_train, imgs_mask_train = mydata.load_train_data()
print(imgs_mask_train.shape)
print('Loading done')
model_checkpoint = ModelCheckpoint('../drive/checkpoints/instru.hdf5', monitor='loss',verbose=1, save_best_only=True)
instru.load_weights('../drive/checkpoints/instru.hdf5')
instru.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
imgs_test=imgs_train[:30]
imgs_mask_test = instru.predict(imgs_test, batch_size=1, verbose=1)
#print(imgs_test)
for i in range(0,imgs_test.shape[0]):
	print(imgs_test[i].shape)
	img=array_to_img(imgs_test[i],scale=True)
	img.save("../drive/Instrument_Segmentation/results/%d_img.jpg"%(i))

	#img=np.reshape(argmax(imgs_mask_test[i],axis=2),(1024,1280,1))
	img=array_to_img(imgs_mask_test[i])
	img.save("../drive/Instrument_Segmentation/results/%d_mask.jpg"%(i))
	#img=np.reshape(argmax(imgs_mask_train[i],axis=2),(1024,1280,1))
	img=array_to_img(imgs_mask_train[i])
	img.save("../drive/Instrument_Segmentation/results/%d_mask_org.jpg"%(i))

