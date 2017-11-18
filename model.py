# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 19:53:27 2017

@author: aamir
"""

import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import random


images = []
measurements = []

#Open and read training data csv file
lines = []
with open('./data-collected/data-laps/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#Read training images and steering angle labels in to lists
zero_steering_weight = 1  #wieght for input images with zero steering angle
for line in lines:
    rand = random.random()
    measurement = float(line[3])
    #Only use a fraction of the images with zero steering angle to balance training data
    if ((measurement == 0) and (rand < zero_steering_weight)) or (measurement != 0):  
        for img_idx in range(3):
            file_path = line[img_idx]
            file_name = file_path.split("\\")
            current_path = './data-collected/data-laps/IMG/' + file_name[-1]
            image = cv2.imread(current_path)
            image_flipped = np.fliplr(image)
            images.extend([image,image_flipped])
        #Apply steering angle correction for images from left and right cameras
        correction = [0,0.08,-0.06]    
        for corr_idx in range(len(correction)):
            steering_angle = float(line[3]) + correction[corr_idx]
            measurements.extend([steering_angle, -steering_angle])

   
#Convert image and measurement lists into numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

n_images = X_train.shape[0]
print(n_images)

#Display sample training images
for dummy_idx in range(1):
    idx = 0
    
    #Display original image
    image = X_train[idx]
    plt.figure()
    plt.imshow(image)
    plt.show()
    
    #Display left and right images and steering angle
    left_image = X_train[idx+2]
    right_image = X_train[idx+4]
    plt.figure()
    plt.imshow(left_image)
    plt.show()
    plt.figure()
    plt.imshow(right_image)
    plt.show()
    print('center cam steering angle:',y_train[idx])
    print('left cam steering angle:',y_train[idx+2])
    print('right cam steering angle:',y_train[idx+4])
    
    #flip image
    image = np.fliplr(image)
    plt.imshow(image)
    plt.show()
    
    #crop image
    image = image[50:140,:]
    plt.imshow(image)
    plt.show()

#Plot histogram
plt.figure()
plt.hist(y_train)
plt.show()

from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,Cropping2D,Lambda,Dropout,Activation

#Initialize model
model = Sequential()

#Preprocessing layers: Normlize and crop image
model.add(Lambda(lambda x:(x/255.0) - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0)),data_format='channels_first'))

#Add convolutional layers with relu activation
model.add(Conv2D(24,5,strides=(2,2),padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(36,5,strides=(2,2),padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(48,5,strides=(2,2),padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(64,3,strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(64,3,strides=(1,1),padding='valid'))
model.add(Activation('relu'))

#Add fully connected and dropoutlayers layers
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))

#Output layer
model.add(Dense(1))

#Train model using adam optimizer and mse loss function
model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

#Save model
model.save('model.h5')


    
    
