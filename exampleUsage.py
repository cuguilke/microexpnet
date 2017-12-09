'''
Title           :exampleUsage.py
Description     :Example usage of the MicroExpNet
Author          :Ilke Cugu & Eren Sener & Emre Akbas
Date Created    :20171210
Date Modified   :20171210
version         :1.0
python_version  :2.7.11
'''
from __future__ import print_function
from time import gmtime, strftime 
from MicroExpNet import *
import tensorflow as tf
import numpy as np
import cv2
import sys
import os

labels = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

# Import the xml files of frontal face and eye
face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml') 

def detectFaces(img):
	# Convertinto grayscale since it works with grayscale images
	gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detect the face
	faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

	if len(faces):
		return faces[0] 
	else:
		return [-13, -13, -13, -13]

# Detects the face and eliminates the rest and resizes the result img
def segmentFace(inputFile, imgXdim, imgYdim):
	# Read the image
	img = cv2.imread(inputFile)  

	# Detect the face
	(p,q,r,s) = detectFaces(img)

	# Return the whole image if it failed to detect the face
	if p != -13: 
		img = img[q:q+s, p:p+r]

	# Crop & resize the image
	img = cv2.resize(img, (256, 256)) 	
	img = img[32:256, 32:256]
	img = cv2.resize(img, (imgXdim, imgYdim)) 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img

def get_time():
	return strftime("%a, %d %b %Y %X", gmtime())


if __name__ == '__main__':
	# Static parameters
	imgXdim = 84
	imgYdim = 84
	nInput = imgXdim*imgYdim # Since RGB is transformed to Grayscale

	if len(sys.argv) != 3:
		print("Usage: python exampleUsage.py <imagePath> <modelDir>")
		print("In detail:")
		print("		imagePath	: Absolute path of the input image")
		print("		modelDir	: Absolute path of the model directory")
		print("Example: python exampleUsage.py neo.jpeg ./Models/OuluCASIA/")
	else:
		# Dynamic parameters
		imagePath = str(sys.argv[1])
		modelDir = str(sys.argv[2])

		# Deploy images and their labels
		print("[" + get_time() + "] " + "Preparation part is completed.")
		print("[" + get_time() + "] " + "Initializing placeholders...")

		# Read the image
		image = segmentFace(imagePath, imgXdim, imgYdim)
		testX = np.reshape(image, (1, imgXdim*imgYdim))
		testX = testX.astype(np.float32)

		# tf Graph input
		x = tf.placeholder(tf.float32, shape=[None, nInput])

		# Construct model
		classifier = MicroExpNet(x)

		# Deploy weights and biases for the model saver
		weights_biases_deployer = tf.train.Saver({"wc1": classifier.w["wc1"], \
											"wc2": classifier.w["wc2"], \
											"wfc": classifier.w["wfc"], \
											"wo": classifier.w["out"],   \
											"bc1": classifier.b["bc1"], \
											"bc2": classifier.b["bc2"], \
											"bfc": classifier.b["bfc"], \
											"bo": classifier.b["out"]})

		with tf.Session() as sess:
			# Initializing the variables
			sess.run(tf.global_variables_initializer())
			print("[" + get_time() + "] " + "Testing is started...")
			weights_biases_deployer.restore(sess, tf.train.latest_checkpoint(modelDir))
			print("[" + get_time() + "] Weights & Biases are restored.")				

			predictions = sess.run([classifier.pred], feed_dict={x: testX})
			argmax = np.argmax(predictions)
			print("[" + get_time() + "] Emotion: " + labels[argmax])