'''
Title           :trainCandidates.py
Description     :This script trains Candidate nets, plots learning curves 
				 and saves corresponding Tensorflow models 
Author          :Ilke Cugu & Eren Sener
Date Created    :28-04-2017
Date Modified   :09-06-2019
version         :1.3
python_version  :2.7.11
'''

from __future__ import print_function
from time import gmtime, strftime
from Preprocessing import *
from CandidateExpNet_v import *
from CandidateExpNet_p1 import *
from CandidateExpNet_p2 import *
from CandidateExpNet_p12 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import cv2
import sys
import os

if __name__ == '__main__':
	# Static parameters  
	imgXdim = 84
	imgYdim = 84
	nInput = imgXdim*imgYdim # Since RGB is transformed to Grayscale
	nClasses = 8
	dropout = 0.5
	batchSize = 64
	#learningRate = 1e-04 
	stepSize = 50000
	epochs = 1000
	testStep = 20
	displayStep = 20	
	'''
		mode					: "-v" 		-> CandidateExpNet_v, 
								  "-p1" 	-> CandidateExpNet_p1
								  "-p2" 	-> CandidateExpNet_p2
								  "-p12" 	-> CandidateExpNet_p12
		valSet					: Index of the chosen test batch (10 batches in total) or file path of the test labels
		labelPath				: Absolute path of the label file
		outputGraphName			: Name of the learning curve graph
		outputModelName			: Name of the Tensorflow model file
		squeezeCoefficient		: Model compression parameter
 	'''
	if len(sys.argv) != 8:
		print("Usage: python trainCandidates.py <mode> <valSet> <labelPath> <outputGraphName> <outputModelName> <learningRate> <squeezeCoefficient>")
	else:
		# Dynamic parameters
		mode = str(sys.argv[1])
		if sys.argv[2].isdigit():
			valSet = int(sys.argv[2])
		else:
			valSet = str(sys.argv[2])
		labelPath = str(sys.argv[3])
		outputGraphName = str(sys.argv[4])
		outputModelName = str(sys.argv[5])
		learningRate = float(sys.argv[6])
		squeezeCoefficient = int(sys.argv[7])
		if mode == "-v":
			print("[" + get_time() + "] " + "Mode: CandidateExpNet_v Training")
		elif mode == "-p1":
			print("[" + get_time() + "] " + "Mode: CandidateExpNet_p1 Training")
		elif mode == "-p2":
			print("[" + get_time() + "] " + "Mode: CandidateExpNet_p2 Training") 
		else:
			print("[" + get_time() + "] " + "Mode: CandidateExpNet_p12 Training")
		
		# Deploy images and their labels
		print("[" + get_time() + "] " + "Deploying images...")
		trainX, trainY, teacherLogits = deployImages(labelPath, None)
		
		# Produce one-hot labels
		print("[" + get_time() + "] " + "Producing one-hot labels...")
		trainY = produceOneHot(trainY, nClasses)
		
		print("[" + get_time() + "] " + "Start training for val[" + str(valSet) + "]")
		
		print("[" + get_time() + "] " + "Initializing batches...")
		batches = []
		test_batches = []
		if type(valSet) == type("str"):
			testX, testY, _ = deployImages(valSet, None)
			testY = produceOneHot(testY, nClasses)
			batches.extend(produceBatch(trainX, trainY, teacherLogits, batchSize))
			test_batches.extend(produceBatch(testX, testY, None, batchSize))
		else:
			# Produce 10 folds for training & validation
			folds = produce10foldCrossVal(trainX, trainY, teacherLogits, labelPath)

			for i in range(10):
				if i != valSet:
					batches.extend(produceBatch(folds[i]['x'], folds[i]['y'], folds[i]['teacherLogits'], batchSize))
				else:
					test_batches.extend(produceBatch(folds[i]['x'], folds[i]['y'], folds[i]['teacherLogits'], batchSize))
		
		print("[" + get_time() + "] " + "Initializing placeholders...")
		
		# tf Graph input
		x = tf.placeholder(tf.float32, shape=[None, nInput]) 
		lr = tf.placeholder(tf.float32) 
		keepProb = tf.placeholder(tf.float32)
		y = tf.placeholder(tf.int32, shape=[None, nClasses])
		
		# Loss values for plotting
		train_loss_vals = []
		train_acc_vals = []
		train_iter_num = []
		test_loss_vals = []
		test_acc_vals = []
		test_iter_num = []
		fin_accuracy = 0
		classifier = None
		
		# Construct model
		if mode == "-v":	
			classifier = CandidateExpNet_v(x, y, lr, nClasses, imgXdim, imgYdim, batchSize, keepProb, squeezeCoefficient)
		elif mode == "-p1":	
			classifier = CandidateExpNet_p1(x, y, lr, nClasses, imgXdim, imgYdim, batchSize, keepProb, squeezeCoefficient)
		elif mode == "-p2":	
			classifier = CandidateExpNet_p2(x, y, lr, nClasses, imgXdim, imgYdim, batchSize, keepProb, squeezeCoefficient)
		else:	
			classifier = CandidateExpNet_p12(x, y, lr, nClasses, imgXdim, imgYdim, batchSize, keepProb, squeezeCoefficient)

		# Deploy weights and biases for the model saver
		model_saver = tf.train.Saver()
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
			print("[" + get_time() + "] " + "Training is started...")
			step = 0
			# Keep training until each max iterations
			while step <= epochs:
				total_batch = len(batches)
				total_test_batch = len(test_batches) 
				for i in range(total_batch):
					batch_x = batches[i]['x']
					batch_y = batches[i]['y']
					# Run optimization op (backprop)
					sess.run(classifier.optimizer, feed_dict={x: batch_x, y: batch_y, lr: learningRate, keepProb: dropout})
				if step % displayStep == 0:
					avg_cost = 0
					avg_perf = 0
					for i in range(total_batch):
						batch_x = batches[i]['x']
						batch_y = batches[i]['y']
						c, p = sess.run([classifier.cost, classifier.accuracy], feed_dict={x: batch_x, y: batch_y, lr: learningRate, keepProb: 1.0})
						avg_cost += c 
						avg_perf += p
					avg_cost /= float(total_batch)
					avg_perf /= float(total_batch)
					train_loss_vals.append(avg_cost)
					train_acc_vals.append(avg_perf)
					train_iter_num.append(step)
					print("[" + get_time() + "] [Iter " + str(step) + "] Training Loss: " + \
							"{:.6f}".format(avg_cost) + " Training Accuracy: " + "{:.5f}".format(avg_perf))
					if avg_cost < -1:
						break
				if step % testStep == 0:
					avg_cost = 0
					fin_accuracy = 0
					for i in range(total_test_batch):
						testX = test_batches[i]['x']
						testY = test_batches[i]['y']					
						c, f = sess.run([classifier.cost, classifier.accuracy], feed_dict={x: testX, y: testY, lr: learningRate, keepProb: 1.0})
						avg_cost += c 
						fin_accuracy += f
					avg_cost /= float(total_test_batch)
					fin_accuracy /= float(total_test_batch)
					test_loss_vals.append(avg_cost)
					test_acc_vals.append(fin_accuracy)
					test_iter_num.append(step)
					print("[" + get_time() + "] [Iter " + str(step) + "] Testing Loss: " + \
							"{:.6f}".format(avg_cost) + " Testing Accuracy: " + "{:.5f}".format(fin_accuracy))						
				if step % stepSize == 0:
					learningRate /= 10
				step += 1
			model_saver.save(sess, outputModelName)
			print("[" + get_time() + "] [Iter " + str(step) + "] Weights & Biases are saved.")
		
		# Print final accuracy independent of the mode
		print ("[" + get_time() + "] Test Accuracy: " + str(fin_accuracy))
		print ("[" + get_time() + "] Training for val[" + str(valSet) + "] is completed.")
			
		# Starting building the learning curve graph	
		fig, ax1 = plt.subplots()

		# Plotting training and test losses
		train_loss, = ax1.plot(train_iter_num, train_loss_vals, color='red',  alpha=.5)
		test_loss, = ax1.plot(test_iter_num, test_loss_vals, linewidth=2, color='green')
		ax1.set_xlabel('Epochs', fontsize=15)
		ax1.set_ylabel('Loss', fontsize=15)
		ax1.tick_params(labelsize=15)

		# Plotting test accuracy
		ax2 = ax1.twinx()
		test_accuracy, = ax2.plot(test_iter_num, test_acc_vals, linewidth=2, color='blue')
		train_accuracy, = ax2.plot(train_iter_num, train_acc_vals, linewidth=1, color='orange')
		ax2.set_ylim(ymin=0, ymax=1)
		ax2.set_ylabel('Accuracy', fontsize=15)
		ax2.tick_params(labelsize=15)
			
		# Adding legend
		plt.legend([train_loss, test_loss, test_accuracy, train_accuracy], ['Training Loss', 'Test Loss', 'Test Accuracy', 'Training Accuracy'],  bbox_to_anchor=(1, 0.8))
		plt.title('Learning Curve', fontsize=18)
		
		# Saving learning curve
		plt.savefig(outputGraphName)	
