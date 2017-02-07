#**Behavioral Cloning** 
---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py checkpoints/model.json
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 16-17 and model function) 

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the preprocess_image function(line 58).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 29). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 97-98). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 31).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used only left and right camera images to train because those images have enough information. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA model and VGG16 model. I stack five convolution layers with max pooling between them. Then I applied couple of dense layers with ELU activation function.

To combat the overfitting, I add drop out layer after each dense layers.
Also, I checked some training epochs and found that validation loss was well converged.

The final step was to run the simulator to see how well the car was driving around track one. 
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
My model could drive the track one at least 1 hour.


####2. Final Model Architecture

The final model architecture (model.py lines 19-32) consisted of a convolution neural network with the following layers and layer sizes.

convolutional layers: 32 - 32 - 32 - 64 - 128 (3x3 filter size each)
dense layers: 1024 - 512 - 256 - 1


####3. Creation of the Training Set & Training Process

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used training data from Udacity for training the model. The validation set helped determine if the model was over or under fitting. 8 epochs was good enough to train the model to be able to drive track one.