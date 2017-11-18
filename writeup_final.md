#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

For my model I decided to start with NVIDIA's self driving car model as this model has been show to work on real life driving data which I hoped would translate well to the simulator. I added dropout and relu layers to prevent overfitting and to add non-linearity to my model. 

The NVIDIA model used 5 convolutional layers and 3 fully connected layers. The convolutional layers act as feature extractors while the fully connected layers act as controllers for the steering angle.

####2. Attempts to reduce overfitting in the model

In order to help reduce over fitting I added dropout layers after the first and second fully connected layers. Based on the fact that the loss for the training and validation set was very small I concluded that my model was suffering from overf itting.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

####4. Appropriate training data

My training data consisted of 5 laps of forward and reverse center lane driving. I augmented the data by flipping the images and also used the left and right camera images to add corrections to the steering angle.

###Model Architecture and Training Strategy

####1. Solution Design Approach

I first implement the basic NVIDIA model and tested it performance on the provided dataset. While I was able to get the car to drive in a straight line I was not able to make it around the first curve.

I then collected my own data consisting of 5 laps of forward and reverse center lane driving. I flipped the images to augment the data set. When I trained using this data the car was able to make it around the first turn but still seemed to be heavily biased towards the right of the track and straight driving.

In order to address this I decided to balance the data distribution by randomly discarding a significant chunk of the zero steering angle data. This greatly improved performance of the model on curves. 

The car was still having trouble recovering once it went off center and keeping itself in the center of the track. In order to address this without actually collecting recovery data I used the left and right camera images to add corrections to the steering angle. This help keep the car centered on the track and greatly improved recovery performance. Tuning the steering angle corrections needed to achieve took significant tuning effort, but made all the difference.

####2. Final Model Architecture

The final model architecture was very similar the NVIDIA model with addition of dropout and relu layers:

**Convolutional layers with relu activation**

`model.add(Conv2D(24,5,strides=(2,2),padding='valid'))`
`model.add(Activation('relu'))`
`model.add(Conv2D(36,5,strides=(2,2),padding='valid'))`
`model.add(Activation('relu'))`
`model.add(Conv2D(48,5,strides=(2,2),padding='valid'))`
`model.add(Activation('relu'))`
`model.add(Conv2D(64,3,strides=(1,1),padding='valid'))`
`model.add(Activation('relu'))`
`model.add(Conv2D(64,3,strides=(1,1),padding='valid'))`
`model.add(Activation('relu'))`

**Fully connected and dropout layers**

`model.add(Flatten())`
`model.add(Activation('relu'))`
`model.add(Dropout(0.5))`
`model.add(Dense(1164))`
`model.add(Activation('relu'))`
`model.add(Dropout(0.5))`
`model.add(Dense(100))`
`model.add(Activation('relu'))`
`model.add(Dense(50))`
`model.add(Activation('relu'))`
`model.add(Dense(10))`

**Output layer**

`model.add(Dense(1))`



####3. Creation of the Training Set & Training Process

I first recorded 5 laps of forward and reverse center lane driving data. Here is an example images from the training set:

![driving_img1](.\sample-images\driving_img1.png)

The image was flipped to augment the data set and prevent it from being biased in favor of a certain track orientation. Here is the flipped image:

![driving_img1_flip](.\sample-images\driving_img1_flip.png)

I used the left and right camera images in order to apply steering corrections for recovery from the sides of the track and to keep the car centered. Here are examples of left and right camera images corresponding to the above example image:

![steering_correction](.\sample-images\steering_correction.png)



![driving_img1_left](.\sample-images\driving_img1_left.png)

![driving_img1_right](.\sample-images\driving_img1_right.png)

After the collection process, I had 59578 number of data points. After I discarded a large chunk of the straight line driving data I had 18972 data points of actual training data. I finally randomly shuffled the data set and put 20% of the data into a validation set. 

Here are distributions before and after discarding the straight line data:

![dist_1](.\sample-images\dist_1.png)



![dist_2](.\sample-images\dist_2.png)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as there was significant decrease in the loss beyond 2 epochs, but there was the risk of overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.