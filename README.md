# huigb-ALPR-chinese

# Chinese license plate recognition in Unscontrained Scenarios

## Introduction

This repository is about Chinese (mainland) license plate recognition sample, based on the Darknet framework and Yolo algorithm.

## Requirements

In order to easily run the code, You must install the environment on which Darknet runs. The Darknet framework is self-contained in the "darknet" folder and must be compiled before running the tests. To build Darknet just type "make" in "darknet" folder:

```shellscript
$ cd darknet && make
```

**The current version was tested in an Ubuntu 14.04 machine, OpenCV 2.4.9, NumPy 1.14 and Python 2.7.**

## Get Models

There are three models, vehicle detection model, license plate detection model, license plate character recognition model, they all in lp_net folder.
All model index paths are added to code, So run test without add path to models.

## Running a simple test
Our ALPR start is very simple, only run follow command.
```shellscript
$ ./darknet
```

## Training the LP detector

To train the LP detector network from scratch, or fine-tuning it for new samples, you can use the train command. In folder mydataset there are 3 annotated samples which are used just for demonstration purposes. To correctly reproduce our experiments, this folder must be filled with all the annotations provided in the training set, and their respective images transferred from the original datasets.


## A word on GPU and CPU

We know that not everyone has an NVIDIA card available, and sometimes it is cumbersome to properly configure CUDA. Thus, we opted to set the Darknet makefile to use CPU as default instead of GPU to favor an easy execution for most people instead of a fast performance. Therefore, the vehicle detection and OCR will be pretty slow. If you want to accelerate them, please edit the Darknet makefile variables to use GPU.
# huigb-ALPR-chinese
# huigb-ALPR-chinese
