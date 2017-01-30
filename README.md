Author: Sze Wun Wong
Title: CarND-Behavioral-Cloning
Date Created: 1-21-2017
Last Updated: 1-21-2017

Table of Content:
------------------------
1) Introduction
2) Approach 
3) Data post-processing
4) 
5)



Introduction:
------------------------

The goal of this behavioral cloning project is to train a car to drive autonomously on a track. A simulator was created by Udacity to allow user to manually drive a car around a track and record the information of the manual driving. The information that this simulator is providing include: images from left camera, images from center camera, image from right camera, steer angle, throttle, brake, and speed. These information will then be feed into a Convolution Neural Network for training and output a steering angle to allow the car to drive autonomously.

Data Collection
------------------------
The first step for this project is to collect data that can be feed into the Convolution Neural Network for training. After playing around with the controls in the simulator, I begin to record data from my manual drive. I circled around the track 5 times, where the goal for the first 3 laps is to stay in the center of the track at all time. Then for the last 2 laps, I weave out toward the border line and immediately head back to the center (once weaving toward the left and once to the right). It is important to do this procedure so that the model can not only learn how to stay in the middle of the track but also know how to react when the car is weaving out toward the left or right border. 

Data Post Processing
------------------------
When recording with the simulator, the simulator return images with size (), however, in looking at the images, there are many unecessary information that can be removed. The feature that is most important from the image is the line marking. Therefore, I removed the upper portion of the image that shows the sky and only retain the portion that shows the line marking and the road. This can not only reduce the dimension of the image but also increase the training time. 