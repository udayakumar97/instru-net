from instru_attention import InstruAttention
from keras.layers import Softmax,Activation
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam,SGD
from datagen import DataGenerator,TestDataGenerator,get_trainval_list
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy,sparse_categorical_accuracy
import numpy as np
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import math
import time
from datetime import datetime
from pytz import timezone
import memory_saving_gradients
import preprocessing
from PIL import Image
from keras.preprocessing.image import array_to_img,load_img,img_to_array

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.95
K.tensorflow_backend.set_session(tf.Session(config=config))
K.__dict__["gradients"] = memory_saving_gradients.gradients_memory

def mean_iou(annotation, logits):
   gt=tf.reshape(annotation,[-1])
   pred_classes=tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)
   preds_flat = tf.reshape(pred_classes, [-1])
   score, up_opt = tf.metrics.mean_iou(gt, preds_flat, num_classes)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

def cce(annotation,logits):
  raw_prediction=tf.reshape(logits,[-1,num_classes])
  gt=tf.reshape(annotation,[-1])
  loss=tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=raw_prediction,labels=gt,name="entropy")))
  return loss

def total_loss(annotation,logits):
  return (cce(annotation,logits)+1-mean_iou)/2

def pixel_acc(annotation,logits):
  gt=tf.reshape(annotation,[-1])
  pred_classes=tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)
  preds_flat = tf.reshape(pred_classes, [-1])
  score,up_opt= tf.metrics.accuracy(gt, preds_flat)
  K.get_session().run(tf.local_variables_initializer())
  with tf.control_dependencies([up_opt]):
      score = tf.identity(score)
  return score

class SaveModel(Callback):
    def on_train_begin(self,logs={}):
        self.epoch=epochs_passed    
    def on_epoch_end(self, batch, logs={}):
        train_model.save_model(model_name)
        self.epoch+=1
        print('epoch:',self.epoch,' time:',datetime.now(timezone('Asia/Kolkata')).strftime('"%Y-%m-%d %H:%M:%S %Z%z"'))

#Change only lr after 10 epochs
epochs_passed=0
ptype=0
model_name='train_instru_attention_binary'
lr1=0.0001
lr2=0.00001
lr=lr1


if ptype==0:
  num_classes=2
elif ptype==1:
  num_classes=4
else:
  num_classes=8
model=InstruAttention(num_classes)
train_model=Model(input=model.input,output=model.output)

train_model.compile(optimizer = Adam(lr), loss = total_loss,metrics=[pixel_acc,mean_iou])
ROOT='datasets/cropped_train/'

train_img_IDs,train_label_IDs,val_img_IDs,val_label_IDs=get_trainval_list(ptype)

params = {'height': 512,
          'width': 512,
          'batch_size': 2,
          'n_channels': 3,
          'ROOT': ROOT,
          'shuffle': True,
          'is_training':True,
          'ptype':ptype}
training_generator = DataGenerator(train_img_IDs,train_label_IDs **params)
params = {'height': 512,
          'width': 512,
          'batch_size': 2,
          'n_channels': 3,
          'ROOT': ROOT,
          'shuffle': False,
          'is_training':False,
          'ptype':ptype}
validation_generator = DataGenerator(val_img_IDs,val_label_IDs **params)
#train_model=load_model(model_name)

save_model=SaveModel()
callbacks_list = [save_model]


train_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    callbacks=callbacks_list,
                    epochs=10)
train_model.save_model(model_name)
'''
ROOT='datasets/test/VOCdevkit/VOC2012'
list_IDs_test = [line.rstrip('\n') for line in open(ROOT+'/ImageSets/Segmentation/test.txt')]
params = {'height': 512,
          'width': 512,
          'batch_size': 1,
          'n_channels': 3,
          'ROOT': ROOT}

#mask=preprocessing.decode_labels(mask.astype(np.uint8))

for i in range(len(list_IDs_test)):
  print('Going to predict')
  filename=list_IDs_test[i]
  ID=list_IDs_test[i]
  img=load_img(ROOT+'/JPEGImages/'+ID+'.jpg',target_size=(512,512),interpolation='bilinear')
  with tf.gfile.Open('../drive/test/masks_GAU_SAM/'+filename+'_org.jpg',mode='w') as f:
    img.save(f)
  img=img_to_array(img)/255.0
  img=np.reshape(img,(1,512,512,3))
  mask=train_model.predict(img,batch_size=1)
  print('Prediction done')
  print(mask.shape)
  img=np.reshape(np.argmax(mask,axis=3),(1,512,512,1))
  img=preprocessing.decode_labels(img)
  img=np.reshape(img,(512,512,3))
  pil_image = array_to_img(img)
  with tf.gfile.Open('../drive/test/masks_GAU_SAM/'+filename+'.png',mode='w') as f:
    pil_image.save(f, 'PNG')
                  
'''
#deeplab_model_org.load_weights('../drive/checkpoints/deeplab.hdf5')
'''
mydata = dataProcess(512, 512)
imgs_train, imgs_mask_train = mydata.load_train_data()
print(imgs_mask_train.shape)
print('Loading done')
imgs_test=deeplab_model.predic(imgs_train,batch_size=1,verbose=1)
for i in range(0,imgs_test.shape[0]):
	print(imgs_test[i].shape)
	img=array_to_img(imgs_test[i])
	img.save('../drive/results/%d_img.jpg'%(i))
	img=array_to_img(imgs_train[i])
	img.save('../drive/results/%d_img_org.jpg'%(i))
	img=array_to_img(imgs_mask_train[i])
	img.save('../drive/results/%d_img_mask.jpg'%(i))
'''
#plot_model(deeplab_model, to_file='deeplabDUC_CAB.png',show_shapes=True)