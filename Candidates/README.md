This directory contains codes of candidate networks for facial expression recognition.

You can start using these models either:
  a) using the sample training script we provide
  b) using direct function calls which are explained below

## Sample Script
trainCandidates is a standalone Python script; you can control its behavior by passing various command-line arguments.

**Usage**

`python trainCandidates.py <mode> <valSet> <labelPath> <outputGraphName> <outputModelName> <learningRate> <squeezeCoefficient>`
  
**Arguments**
  * mode: 
  	* "-v"   -> CandidateExpNet_v, 
	* "-p1"  -> CandidateExpNet_p1
	* "-p2"  -> CandidateExpNet_p2
	* "-p12" -> CandidateExpNet_p12
  * valSet: Index of the chosen test batch (10 batches in total) or file path of the test labels
  * labelPath: Absolute path of the label file
  * outputGraphName: Name of the learning curve graph
  * outputModelName: Name of the Tensorflow model file
  * squeezeCoefficient: Model compression parameter

## API
**CandidateExpNet_v(x, y, lr, nClasses, imgXdim, imgYdim, batchSize, keepProb, squeezeCoefficient)**

**CandidateExpNet_p1(x, y, lr, nClasses, imgXdim, imgYdim, batchSize, keepProb, squeezeCoefficient)**

**CandidateExpNet_p2(x, y, lr, nClasses, imgXdim, imgYdim, batchSize, keepProb, squeezeCoefficient)**

**CandidateExpNet_p12(x, y, lr, nClasses, imgXdim, imgYdim, batchSize, keepProb, squeezeCoefficient)** 

These classes are used to build each candidate networks explained in the paper.

**Parameters**
  - x: Tensorflow placeholder for input images 
  - y: Tensorflow placeholder for one-hot labels
  - lr: Learning rate (default: 1e-04)
  - nClasses: Number of emotion classes (default: 8)
  - imgXdim: Dimension of the image (default: 84)
  - imgYdim: Dimension of the image (default: 84)
  - batchSize: Batch size (default: 64)
  - keepProb: Dropput (default: 0.5)
  - squeezeCoefficient: Hyperparameter to tune the size of the network (default: 1) 
    * 1 -> M
    * 4 -> S
    * 8 -> XS
    * 16 -> XXS
    
