from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
#import cv2
#from libtiff import TIFF

class dataProcess(object):

	def __init__(self, out_rows, out_cols, dataset_path = "../deform/train", problem_type='binary',img_type='jpg',label_type='png',test_path = "../test", npy_path = "../npydata"):

		"""
		
		"""

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.dataset_path = dataset_path
		self.problem_type = problem_type
		self.img_type = img_type
		self.label_type = label_type
		self.test_path = test_path
		self.npy_path = npy_path

	def create_train_data(self):
		i = 0
		imgdatas = np.ndarray((225*2,self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((225*2,self.out_rows,self.out_cols,1), dtype=np.uint8)
		for j in range(1,3):
			data_path=self.dataset_path+'/instrument_dataset_'+str(j)+'/images'
			label_path=self.dataset_path+'/instrument_dataset_'+str(j)+'/binary_masks'
			print(data_path)
			imgs = glob.glob(data_path+"/*."+self.img_type)
			print(imgs)
			print(len(imgs))
			for imgname in imgs:
				midname = imgname[imgname.rindex("/")+1:-3]
				img = load_img(data_path + "/" + midname+self.img_type)
				label = load_img(label_path + "/" + midname+self.label_type,grayscale = True)
				img = img_to_array(img)
				label = img_to_array(label)
				#img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
				#label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
				#img = np.array([img])
				#label = np.array([label])
				imgdatas[i] = img
				imglabels[i] = label
				if i % 100 == 0:
					print('Done: {0}/{1} images'.format(i, 8*225))
				i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train_binary.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train_binary.npy', imglabels)
		print('Saving to .npy files done.')

	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/image/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		#imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.test_path + "/image/" + midname,grayscale = True)
			#label = load_img(self.test_path + "/label/" + midname,grayscale = True)
			img = img_to_array(img)
			#label = img_to_array(label)
			#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			imgdatas[i] = img
			#imglabels[i] = label
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		#np.save(self.npy_path + '/imgs_mask_test.npy', imglabels)
		print('Saving to .npy files done.')

	def load_train_data(self):
		print('load train images...')
		imgs_train = np.load(self.npy_path+"/imgs_train_binary.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train_binary.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 127.5
		imgs_train-=1
		#mean = imgs_train.mean(axis = 0)
		#imgs_train -= mean	
		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train,imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		#imgs_mask_test = np.load(self.npy_path+"/imgs_mask_test.npy")
		imgs_test = imgs_test.astype('float32')
		#imgs_mask_test = imgs_mask_test.astype('float32')
		imgs_test /= 255
		#mean = imgs_test.mean(axis = 0)
		#imgs_test -= mean	
		#imgs_mask_test /= 255
		#imgs_mask_test[imgs_mask_test > 0.5] = 1
		#imgs_mask_test[imgs_mask_test <= 0.5] = 0
		return imgs_test#,imgs_mask_test

if __name__ == "__main__":

	#aug = myAugmentation()
	#aug.Augmentation()
	#aug.splitMerge()
	#aug.splitTransform()
	mydata = dataProcess(1024, 1280,dataset_path= "/cropped_train", problem_type='binary', test_path = "../dataset/test")
	mydata.create_train_data()
	#mydata.create_test_data()
	#imgs_train,imgs_mask_train = mydata.load_train_data()
	#print imgs_train.shape,imgs_mask_train.shape
