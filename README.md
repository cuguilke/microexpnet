# MicroExpNet

By [Ilke Cugu](https://cuguilke.github.io/), [Eren Sener](https://scholar.google.com.tr/citations?user=xDRyyxoAAAAJ&hl=en), [Emre Akbas](http://user.ceng.metu.edu.tr/~emre/).

## Table of Contents

1. [Introduction](#introduction)
2. [Citation](#citation)
3. [API](#api)
4. [Models](#models)

## Introduction

MicroExpNet is an extremely small (under 1MB) and fast (1851 FPS on i7 CPU) [TensorFlow](https://www.tensorflow.org/) convolutional neural network model for facial expression recognition (FER) from frontal face images.  This repository contains the codes described in the paper "MicroExpNet: An Extremely Small and Fast Model For Expression Recognition From Face Images" (https://arxiv.org/abs/1711.07011v4).

**Full list of items**
  * MicroExpNet.py: The original source code of the proposed FER model
  * exampleUsage.py: A script to get prediction from a pre-trained MicroExpNet for an unlabeled image
  * Models: Pre-trained MicroExpNet models for CK+ and Oulu-CASIA datasets.
  * Candidates: Candidate networks build in search of a better FER model
  
## Citation

If you use these models in your research, please cite:

```
@inproceedings{cugu2019microexpnet,
  title={MicroExpNet: An Extremely Small and Fast Model For Expression Recognition From Face Images},
  author={Cugu, Ilke and Sener, Eren and Akbas, Emre},
  booktitle={2019 Ninth International Conference on Image Processing Theory, Tools and Applications (IPTA)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
```

## API
**MicroExpNet(x, y, teacherLogits, lr, nClasses, imgXdim, imgYdim, batchSize, keepProb, temperature, lambda_)**

This is the class where the magic happens. Take a look at **exampleUsage.py** for a quick test drive.

**Parameters**
  - x: Tensorflow placeholder for input images 
  - y: Tensorflow placeholder for one-hot labels (default: None -> unlabeled image testing)
  - teacherLogits: Tensorflow placeholder for the logits of the teacher (default: None -> for standalone testing)
  - lr: Learning rate (default: 1e-04)
  - nClasses: Number of emotion classes (default: 8)
  - imgXdim: Dimension of the image (default: 84)
  - imgYdim: Dimension of the image (default: 84)
  - batchSize: Batch size (default: 64)
  - keepProb: Dropout (default: 1)
  - temperature: The hyperparameter to soften the teacher's probability distributions (default: 8)
  - lamba_: Weight of the soft targets (default: 0.5)

## Models

We provide pre-trained MicroExpNet models for both CK+ and Oulu-CASIA.

In addition, one can find sample pre-trained teacher models which are derived from the original [Keras](https://github.com/keras-team/keras) implementation of [Inception_v3](https://keras.io/applications/#inceptionv3):
 * [TeacherExpNet_CK.h5](http://user.ceng.metu.edu.tr/~e1881739/microexpnet/TeacherExpNet_CK.h5)
 * [TeacherExpNet_OuluCASIA.h5](http://user.ceng.metu.edu.tr/~e1881739/microexpnet/TeacherExpNet_OuluCASIA.h5) 
 * [TeacherExpNet.json](http://user.ceng.metu.edu.tr/~e1881739/microexpnet/TeacherExpNet.json) 

**Labels of the both models**

`0: neutral, 1: anger, 2: contempt, 3: disgust, 4: fear, 5: happy, 6: sadness, 7: surprise`
