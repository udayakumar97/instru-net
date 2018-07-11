from keras import backend as K
def IOU_calc(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	smooth=1
	intersection = K.sum(y_true_f * y_pred_f)
	
	return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def IOU_calc_multi(y_true,y_pred, num_classes=4):
	iou=0
	#y_true=K.reshape(y_true,(K.int_shape(y_true)[0],K.int_shape(y_true)[1],K.int_shape(y_true)[2]))
	y_true=K.one_hot(K.cast(y_true,"int32"),4)
	y_pred=K.one_hot(K.cast(y_true,"int32"),4)
	for index in range(num_classes):
		iou+=IOU_calc(y_true[:,:,:,index],y_pred[:,:,:,index])
	return iou

def IOU_calc_loss(y_true, y_pred):
	return -IOU_calc(y_true,y_pred)

def IOU_calc_multi_loss(y_true,y_pred,num_classes=4):
	return -IOU_calc_multi(y_true,y_pred,num_classes)