'''
Title           :Preprocessing.py
Description     :This script contains all functions required for data augmentation & preparation
Author          :Ilke Cugu & Eren Sener
Date Created    :28-04-2017
Date Modified   :09-06-2019
version         :1.1
python_version  :2.7.11
'''

from time import gmtime, strftime
import numpy as np
import cv2

def mirrorImages(img, newDim=84):
	mirror_img0 = img[0:224, 0:224]
	mirror_img0 = cv2.resize(mirror_img0, (newDim, newDim))	
	mirror_img1 = img[0:224, 5:229]
	mirror_img1 = cv2.resize(mirror_img1, (newDim, newDim))	
	mirror_img2 = img[0:224, 32:256]
	mirror_img2 = cv2.resize(mirror_img2, (newDim, newDim))	
	mirror_img3 = img[5:229, 0:224]
	mirror_img3 = cv2.resize(mirror_img3, (newDim, newDim))	
	mirror_img4 = img[32:256, 0:224]
	mirror_img4 = cv2.resize(mirror_img4, (newDim, newDim))	
	mirror_img5 = img[5:229, 5:229]
	mirror_img5 = cv2.resize(mirror_img5, (newDim, newDim))	
	mirror_img6 = img[32:256, 32:256]
	mirror_img6 = cv2.resize(mirror_img6, (newDim, newDim))	
	mirror_img7 = img[5:229, 32:256]
	mirror_img7 = cv2.resize(mirror_img7, (newDim, newDim))	
	return [mirror_img0, mirror_img1, mirror_img2, mirror_img3, mirror_img4, mirror_img5, mirror_img6, mirror_img7]

def deployImages(labelpath, TeacherSoftmaxInputs):
	images = []
	imageNames = []
	labels = np.array([])
	y_values = []
	with open(labelpath, "r") as labelfile:
		for f in labelfile:
			f = f.split(" ")
			filename = f[0]
			imageNames.append(filename)
			#imageNames.append(filename.split('/')[-1])
			label = int(f[1])
			im = cv2.imread(filename) 
			gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  
			mirror_ims = mirrorImages(gray_im)
			for i in mirror_ims:
				gray_im = i.reshape(i.shape[0]*i.shape[1])				
				images.append(gray_im)
				labels = np.append(labels, label)
				if TeacherSoftmaxInputs != None:
					y_values.append(TeacherSoftmaxInputs[filename])	

	return np.array(images), labels, np.array(y_values)
		
def produceOneHot(y, n_classes):
	n = y.shape[0]
	new_y = np.zeros([n, n_classes])
	for i in range(0,n):
		new_y[i][int(y[i])] = 1
	return new_y 

def produce10foldCrossVal(x, y, teacherLogits, labelpath):
	# Separate validation and training sets 
	folds = []
	labels = []
	with open(labelpath, "r") as labelfile:
		for f in labelfile:
			# This line gathers fold IDs (10-fold cross-val) from a label file with a format like:
			#	...
			#	/home/cuguilke/Desktop/CK_new/fold1/n01497344_S028_001_00018.jpeg 1
			#	/home/cuguilke/Desktop/CK_new/fold1/n01448192_S052_001_00045.jpeg 7
			#	/home/cuguilke/Desktop/CK_new/fold1/n01456384_S044_004_00065.jpeg 0
			#	/home/cuguilke/Desktop/CK_new/fold1/n01489152_S035_006_00021.jpeg 5
			#	...
			# Feel free to modify this part with your own way of indicating fold IDs
			labels.append(int(f.split('/')[5][-1]))
	partition_x = []
	partition_y = []
	partition_teacherLogits = [] 
	last_batch_no = 0
	n = len(labels)
	for i in range(n):
		batch_no = labels[i]
		if batch_no != last_batch_no:
			folds.append({'x': np.array(partition_x), 'y': np.array(partition_y), 'teacherLogits': None})
			partition_x = []
			partition_y = []
			partition_teacherLogits = []
		else:
			for j in range(i*8,(i+1)*8): # Since each image is mirrored 
				partition_x.append(x[j])
				partition_y.append(y[j])
				if teacherLogits != None:
					teacherNetOut = teacherLogits[j]
					partition_teacherLogits.append(teacherNetOut)
				else:
					partition_teacherLogits.append([])  
		last_batch_no = batch_no
	if len(partition_x) > 0:
		folds.append({'x': np.array(partition_x), 'y': np.array(partition_y), 'teacherLogits': None})	
	return folds

def produceBatch(x, y, teacherLogits, batchSize):
	batches = []
	n = y.shape[0]
	batch_x = []
	batch_y = []
	batch_teacherLogits = []
	for i in range(0,n):
		if i % batchSize == 0 and i != 0:
			batches.append({'x': np.array(batch_x), 'y': np.array(batch_y), 'teacherLogits': np.array(batch_teacherLogits)})
			batch_x = []
			batch_y = []
			batch_teacherLogits = []
		batch_x.append(x[i])
		batch_y.append(y[i])
		if teacherLogits != None:
			batch_teacherLogits.append(teacherLogits[i]) 
		else:
			batch_teacherLogits.append([]) 
	if len(batch_x) > 0:
		batches.append({'x': np.array(batch_x), 'y': np.array(batch_y), 'teacherLogits': np.array(batch_teacherLogits)}) # Add the leftovers as the last batch
	return batches

def get_time():
	return strftime("%a, %d %b %Y %X", gmtime())