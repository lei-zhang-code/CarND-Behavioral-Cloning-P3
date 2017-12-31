# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the following layers (model.py line 97 ~ 114):
* Convolution layer, 32 filters, 3x3 kernel, with 2x2 max-pooling and Relu activation.
* Convolution layer, 64 filters, 3x3 kernel, with 2x2 max-pooling and Relu activation.
* Convolution layer, 128 filters, 3x3 kernel, with 2x2 max-pooling and Relu activation.
* Flatten out layer.
* Dropout layer with 0.5 keep-probability.
* Fully connected layer with 256 outputs and Relu activation.
* Dropout layer with 0.5 keep-probability.
* Fully connected layer with 64 outputs and Relu activation.
* Dropout layer with 0.5 keep-probability.
* Fully connected layer with 1 output.

The data preprocessing includes (model.py line 93 ~ 95):
* Normalize all pixel values in the input image to [-0.5, 0.5] range.
* Crop the top 70 rows and bottom 20 rows of pixels. Since this is the only region that contains the road.
* Downsample the image by 2 folder using Keras's AveragePooling2D layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 107, 110, 113). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 117). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the center camera as well as the two side cameras to train the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the VGG network.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set by 80:20. The network trained very well with also low validation loss. However, the vehicle runs out of track.

This suggest that I don't have enough data to cover the turns. So I collected two more laps of the data.

#### 2. Final Model Architecture

My model consists of the following layers (model.py line 97 ~ 114):
* Convolution layer, 32 filters, 3x3 kernel, with 2x2 max-pooling and Relu activation.
* Convolution layer, 64 filters, 3x3 kernel, with 2x2 max-pooling and Relu activation.
* Convolution layer, 128 filters, 3x3 kernel, with 2x2 max-pooling and Relu activation.
* Flatten out layer.
* Dropout layer with 0.5 keep-probability.
* Fully connected layer with 256 outputs and Relu activation.
* Dropout layer with 0.5 keep-probability.
* Fully connected layer with 64 outputs and Relu activation.
* Dropout layer with 0.5 keep-probability.
* Fully connected layer with 1 output.

The data preprocessing includes (model.py line 93 ~ 95):
* Normalize all pixel values in the input image to [-0.5, 0.5] range.
* Crop the top 70 rows and bottom 20 rows of pixels. Since this is the only region that contains the road.
* Downsample the image by 2 folder using Keras's AveragePooling2D layer.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track one using center lane driving. I finally randomly shuffled the data set and put 20% of the data into a validation set. For the left and right camera images, I added 0.2 and -0.2 steer angle offset, respectively.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the increased validation error after 2 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
