from os.path import join
import glob
files_list = glob.glob('tiny-imagenet-200/train/*/images/*')
label_list={}
lines = [line.rstrip('\n') for line in open('tiny-imagenet-200/wnids.txt')]
for i,item in enumerate(lines):
	label_list[item]=i
with open('list_IDs_train.txt','w') as f1:
	with open('labels_train.txt','w') as f2:
		for item in files_list:
			f1.write("%s\n"%item)
			f2.write("%s\n"%label_list[item[24:33]])

with open('list_IDs_val.txt','w') as f1:
	with open('labels_val.txt','w') as f2:
		for line in open('tiny-imagenet-200/val/val_annotations.txt','r'):
			line.rstrip('\n').split('\t')[0]
			f1.write("%s\n"%('tiny-imagenet-200/val/images/'+line.rstrip('\n').split('\t')[0]))
			f2.write("%s\n"%label_list[line.rstrip('\n').split('\t')[1]])

