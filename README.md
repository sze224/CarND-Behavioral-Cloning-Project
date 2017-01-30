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

The goal of this behavioral cloning project is to train a car to drive autonomously on a track. A simulator was created by Udacity to allow user to manually drive a car around a track and record the information of the manual driving. The information that this simulator is providing include: images from left camera, images from center camera, image from right camera, steer angle, throttle, brake, and speed. These information will then be fed into a Convolution Neural Network for training and output a steering angle to allow the car to drive autonomously.

Data Collection:
------------------------
The first step for this project is to collect data that can be feed into the Convolution Neural Network for training. After playing around with the controls in the simulator, I begin to record data from my manual drive. I circled around the track 7 times and with the goal to mimic how I would drive on an actual road (ie: stay within the drivable area, not drive off the road and turn as smoothly as possible). 

Visualizing Data:
------------------------
Before deciding how the model should be trained, it is really important to visualize the training data. This will determine what type of speical techniques or methods I should utilized when training the model. 

<img width="575" alt="screen shot 2017-01-29 at 8 41 39 pm" src="https://cloud.githubusercontent.com/assets/22971963/22412498/c1e84db2-e663-11e6-8496-9b9f4cf2b952.png">

Taking a first look at the data collected, it is obvious that there are much more data point of 0 deg than any other value. Since the training track has more straight paths than turns, this observation is not surprising.

Another thing to notice from the data is that there are more negative steer angle than positive steer angle. Again this is due to the nature of the training track where there are more left turn than right turn. 

Even though this graph seems simple but it actually contains many important information. I now know that I would need to apply some type of method to avoid the model biasing toward 0 deg and balance the data set to have both negative steer angles and positive steer angles.

Data Post Processing
------------------------
When recording with the simulator, the simulator return images with size of 160,320,3 since this is the size of the simulator window selected. However, in looking at the images, there are many unecessary information that can be removed. 

<img width="510" alt="screen shot 2017-01-29 at 9 05 57 pm" src="https://cloud.githubusercontent.com/assets/22971963/22412767/c1e6e208-e666-11e6-9515-9ab90ebdee7d.png">


The feature that is most important from the image is the line marking. Therefore, I removed the upper portion of the image that shows the sky and only retain the portion that shows the line marking and the road. This can not only reduce the dimension of the image but also increase the training time. 



