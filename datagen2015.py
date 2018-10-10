import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
from keras import backend as K
import random
import tensorflow as tf

def get_trainval_list(ptype):
    train_imgs_list=[]
    train_label_list=[]
    val_imgs_list=[] 
    val_label_list=[]
    last={1:1106,2:1124,3:1123,4:1122}
    ROOT='Segmentation_Robotic_Training/Training/'
    for i in range(1,5):
        for j in range(0,800):
            train_imgs_list.append(ROOT+'Dataset'+str(i)+'/Raw/frame'+str(j)+'.png')
            train_label_list.append(ROOT+'Dataset'+str(i)+'/Masks/frame'+str(j)+'.png')
        
        for j in range(801,last[i]+1):
            val_imgs_list.append(ROOT+'Dataset'+str(i)+'/Raw/frame'+str(j)+'.png')
            val_label_list.append(ROOT+'Dataset'+str(i)+'/Masks/frame'+str(j)+'.png')
    return train_imgs_list,train_label_list,val_imgs_list,val_label_list

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img_IDs, label_IDs, ROOT,batch_size=8, height=480,width=640, n_channels=3,
                 shuffle=False,is_training=False,ptype=0):
        'Initialization'
        self.img_IDs=img_IDs
        self.label_IDs=label_IDs
        self.height=height
        self.width=width
        self.batch_size = batch_size
        self.ROOT=ROOT
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.is_training=is_training
        self.ptype=ptype  #0 is class 1 is instrument 
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_IDs_temp = [self.img_IDs[k] for k in indexes]
        label_IDs_temp=[self.label_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(img_IDs_temp,label_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_IDs_temp,label_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.ndarray((self.batch_size, self.height,self.width, self.n_channels), dtype=np.float32)
        y = np.ndarray((self.batch_size,self.height,self.width,1), dtype=np.uint8)

        # Generate data
        for i in range(len(img_IDs_temp)):
            # Store sample
            
            image=load_img(self.ROOT+img_IDs_temp[i],target_size=(self.height,self.width))
            image=img_to_array(image)

            # Store class
            label=load_img(self.ROOT+label_IDs_temp[i],grayscale=True,target_size=(self.height,self.width))
            label=img_to_array(label)
            if self.is_training:
                
                flip=random.randint(0,1)
                if flip==1:
                    image=np.fliplr(image)
                    label=np.fliplr(label)
                
            X[i,]=image/255.0
            if self.ptype==0:
                label[label<=35]=0
                mask=np.logical_and(label>35,label<=115)
                label[mask]=1
                label[label>115]=2
            else:
                label[label<=35]=0
                label[label>35]=1
            y[i,]=label

        return X, y

class TestDataGenerator(keras.utils.Sequence):
    'Generates test data for Keras'
    def __init__(self, list_IDs,ROOT,batch_size=32, height=512,width=512, n_channels=3):
        'Initialization'
        self.height=height
        self.width=width
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.ROOT=ROOT
        self.n_channels = n_channels
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.ndarray((self.batch_size, self.height,self.width, self.n_channels), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            image=load_img(self.ROOT+'/JPEGImages/'+ID+'.jpg',target_size=(self.height,self.width),interpolation='bilinear')
                
            X[i,]=img_to_array(image)/255.0

        return X