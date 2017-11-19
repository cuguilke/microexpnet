# MicroExpNet

By [Ilke Cugu](https://scholar.google.com.tr/citations?user=v6YG0YEAAAAJ&hl=en), Eren Sener, [Emre Akbas](http://user.ceng.metu.edu.tr/~emre/).

## Table of Contents

1. [Introduction](#introduction)
2. [Citation](#citation)
3. [API](#api)

## Introduction

MicroExpNet is an extremely small CNN written with the [TensorFlow](https://www.tensorflow.org/) library for facial expression recognition (FER) using frontal face images. This repository contains the codes described in the paper "MicroExpNet: An Extremely Small and Fast Model For Expression Recognition From Frontal Face Images" (LINK WILL BE ADDED).

**Full list of items**
  * MicroExpNet.py: The original source code of the proposed FER model
  * Teacher_Logits: Logits of the teacher network for knowledge distillation
  * Candidates: Candidate networks build in search of a better FER model
  
## Citation

If you use these models in your research, please cite:
(BIBTEX WILL BE ADDED)

## API
**MicroExpNet(x, y, teacherLogits, lr, nClasses, imgXdim, imgYdim, batchSize, keepProb, temperature, lambda_)**

This is the class where the magic happens.

**Parameters**
  - x: Tensorflow placeholder for input images 
  - y: Tensorflow placegolder for one-hot labels
  - teacherLogits: Tensorflow placeholder for the logits of the teacher
  - lr: Learning rate (default: 1e-04)
  - nClasses: Number of emotion classes (default: 8)
  - imgXdim: Dimension of the image (default: 84)
  - imgYdim: Dimension of the image (default: 84)
  - batchSize: Batch size (default: 64)
  - keepProb: Dropput (default: 0.5)
  - temperature: The hyperparameter to soften the teacher's probability distributions (default: 8)
  - lamba_: Weight of the soft targets (default: 0.5)
