'''
Title           :CandidateExpNet_p2.py
Description     :Candidate CNN class for emotion recognition with a pooling layer
Author          :Ilke Cugu & Eren Sener
version         :1.0
python_version  :2.7.11
'''
import tensorflow as tf

class CandidateExpNet_p2():
	def __init__(self, x, y, lr=1e-04, nClasses=8, imgXdim=84, imgYdim=84, batchSize=64, keepProb=0.5, squeezeCoefficient=1):
		self.x = x
		self.y = y 
		self.w = {}
		self.b = {}
		self.imgXdim = imgXdim
		self.imgYdim = imgYdim
		self.nClasses = nClasses
		self.batchSize = batchSize 
		self.learningRate = lr 		# Default = 1e-04 
		self.dropout = keepProb 	# 0.5
		self.fcOutSize = 3 * 256 / squeezeCoefficient

		# Initialize parameters randomly and run
		self.initParameters()
		self.output, self.layerInfo = self.run() 
		
		# Define losses and optimizers 
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
		self.pred = tf.nn.softmax(self.output)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost)
		#self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learningRate).minimize(self.cost)
		#self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(self.cost)		

		# Evaluate model 
		self.correct_pred= tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

	def initParameters(self):
		self.w = {
		# 8x8 conv, 1 input channel, 16 outputs
		'wc1': tf.get_variable('wc1', [8, 8, 1, 16], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
		# 4x4 conv, 16 inputs, 32 outputs
		'wc2': tf.get_variable('wc2', [4, 4, 16, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
		# fully connected, 1152 inputs, 768 / squeezeCoefficient outputs
		'wfc': tf.get_variable('wfc', [32*6*6, self.fcOutSize], initializer=tf.contrib.layers.xavier_initializer()),
		# 256 inputs, 8 outputs (class prediction)
		'out': tf.get_variable('wo', [self.fcOutSize, self.nClasses], initializer=tf.contrib.layers.xavier_initializer())
		}
		self.b = {
			'bc1': tf.Variable(tf.random_normal(shape=[16], stddev=0.5), name="bc1"),
			'bc2': tf.Variable(tf.random_normal(shape=[32], stddev=0.5), name="bc2"),
			'bfc': tf.Variable(tf.random_normal(shape=[self.fcOutSize],stddev=0.5), name="bfc"),
			'out': tf.Variable(tf.random_normal(shape=[self.nClasses], stddev=0.5), name="bo")
		}

	def conv2d(self, x, W, b, strides=1):
		# Conv2D wrapper, with bias and relu activation
		x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		return x 

	def ReLU(self, x):
		# ReLU wrapper
		return tf.nn.relu(x)

	def maxpool2d(self, x, k=2):
		# MaxPool2D wrapper
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME') 

	def run(self):
		# Reshape input picture
		# Input is a 1D numpy.array created from grayscale JPEG files
		x = tf.reshape(self.x, shape=[-1, self.imgXdim, self.imgYdim, 1])

		# Convolution Layer
		conv1 = self.conv2d(x, self.w['wc1'], self.b['bc1'], 4) 
		# ReLU Regularization
		conv1_relu = self.ReLU(conv1)

		# Convolution Layer
		conv2 = self.conv2d(conv1_relu, self.w['wc2'], self.b['bc2'], 2) 
		# ReLU Regularization
		conv2_relu = self.ReLU(conv2)
		# Max Pooling
		conv2_pool = self.maxpool2d(conv2_relu)		

		# Fully connected later
		fc1 = tf.reshape(conv2_pool, [-1, self.w['wfc'].get_shape().as_list()[0]])
		fc1 = tf.add(tf.matmul(fc1, self.w['wfc']), self.b['bfc'])
		fc1 = self.ReLU(fc1) 
		fc1 = tf.nn.dropout(fc1, self.dropout)

		# Output, class prediction
		out = tf.add(tf.matmul(fc1, self.w['out']), self.b['out']) 

		# Packing the results
		layerInfo = [conv1, conv1_relu, conv2, conv2_relu, fc1, out]
		
		return out, layerInfo
