Author: Sze Wun Wong
Title: CarND-Behavioral-Cloning
Date Created: 01-21-2017
Last Updated: 02-05-2017

Table of Content:
------------------------
1) Introduction

2) Data Collection

3) Visualizing Data

4) Data Post Processing

5) Generate Train Set

6) Model Architecture/ Parameters

7) Reflection

8) video links

Introduction:
------------------------

The goal of this behavioral cloning project is to train a car to drive autonomously on a track. A simulator was created by Udacity to allow user to manually drive a car around a track and record the information of the manual driving. The information that this simulator is recording include: images from left camera, images from center camera, image from right camera, steer angle, throttle, brake, and speed. For this project, I only utilize the images from left, right and center camera and steer angle. These information will be fed into a Convolution Neural Network for training and the model will output a steering angle to allow the car to drive autonomously.

Data Collection:
------------------------
The first step for this project is to collect data that can be feed into the Convolution Neural Network for training. Udacity provided a set of data that I can train my network on. In looking at the data, most of the data reflect a steering angle of 0, which doesn't truly provide enough data and capture the fact that there are 2 more aggressive turn at this track.

![center_2016_12_01_13_31_15_308](https://cloud.githubusercontent.com/assets/22971963/22635394/3585c626-ebe8-11e6-870c-fda0f931a02f.jpg)
![center_2016_12_01_13_33_10_173](https://cloud.githubusercontent.com/assets/22971963/22635415/70cd45f6-ebe8-11e6-8e2a-578bc0a844d1.jpg)

Therefore, to alleviate this problem, I recorded extra information that aims to execute these more aggressive turn.

Visualizing Data:
------------------------
Before deciding how the model should be trained, it is really important to visualize the training data. This will determine what type of speical techniques or methods I should utilized when training the model. 

<img width="565" alt="screen shot 2017-02-05 at 9 19 31 pm" src="https://cloud.githubusercontent.com/assets/22971963/22635458/d64a2fb6-ebe8-11e6-81e9-7ec976db805a.png">

Taking a first look at the data collected, it is obvious that there are much more data point of 0 deg than any other value. Since the training track has more straight paths than turns, this observation is not surprising.

Another thing to notice from the data is that there are slightly more agressive negative steer angle than positive steer angle. This need to be address in training so the car will learn how to do both aggressive left turn and right turn. 

Even though plotting this graph seems simple but it actually contains many important information. I now know that I would need to apply some type of method to avoid the model biasing toward 0 deg and ensure there are data set to have both aggressive negative and positive steer angle. 

Data Post Processing
------------------------
When recording with the simulator, the simulator return images with size of 160,320,3 since this is the size of the simulator window selected. However, in looking at the images, there are many unecessary information that can be removed. 

<img width="524" alt="screen shot 2017-02-05 at 9 23 47 pm" src="https://cloud.githubusercontent.com/assets/22971963/22635603/8a7eb0d8-ebe9-11e6-9d53-b23ac16cfb6d.png">

The feature that is most important from the image is the line marking. Therefore, I removed the upper portion of the image that shows the sky and only retain the portion that shows the line marking and the road. This can not only reduce the dimension of the image but also decrease the training time. 

<img width="509" alt="screen shot 2017-02-05 at 9 24 00 pm" src="https://cloud.githubusercontent.com/assets/22971963/22635605/8c0bc922-ebe9-11e6-8fae-2ee5161b4296.png">

To futher reduce the dimension, I resized the image to have a size of 60x60x3. I experimented with different value and it seems that resizing this this dimension didn't affect the training of the model. However, it does reduce training time.

<img width="365" alt="screen shot 2017-02-05 at 9 24 05 pm" src="https://cloud.githubusercontent.com/assets/22971963/22635607/8d4a38d2-ebe9-11e6-8909-f1aa3ba2ebf1.png">

Generate Training Set
------------------------
When training the model, it is important to not over fit the model to the training set. To avoid this problem, a logical thing to do is to do image augmentation. This is basically creating more training examples from the existing training examples. From image augmentation, I decided to used the following techniques: randomly translating the image, using left and right camera images,  randomly flipping the image and randomly applying different brightness of the image.

1) The goal of randomly translating the image is to imitate the car being in different positions. Since it is not likely that the car will drive perfectly in the center of road at all times, the car needs to know how it should response when it is at a off position. For horizontal translation, I added a steer bias for each pixal that it is translated. In other words, if the car is close to the edge, it should be able to steering itself back to the drivable portion. For vertical translation, no steer bias are needed. The goal is to mimic what the car will see when it is going uphill or downhill.

2) The goal of using left and right camera is again to address the problem of recovery from off position (edge of the road or close to un driveable area). Again, when using the left and right camera images, it is important to add a bias to the steer angle. It is important to pick a bias that will not only allow the car to stay in the drivable area, but also have a small oscillation.

3) As noted in the "visualizing data" section, we see that there are more data with a aggressive negative steer angle than positive steer angle. It is important to try to balance out the data so the model doesn't overfit. The way that I approach this is to randomly flip the image and negate the steer angle. This way there will be training data from with aggressive negative steer angles and aggressive positive steer angle.

4) Randomly changing the brightness of the image allow the model to know how to react in different type of environment. It is not likely that the track will always be as bright as track 1, therefore it is important for the car to know how to react with the track is dark (just like track 2).

Model Architecture/ Parameters
------------------------
<img width="171" alt="screen shot 2017-01-31 at 7 29 29 pm" src="https://cloud.githubusercontent.com/assets/22971963/22494352/a3bd9f5c-e7eb-11e6-88b4-e987d78c7bb9.png">

The parameters of the model were determined by trial and error. However, this model's basic structure is pattern after the famous VGG net. VGG net is a network of repeating 3x3 convolution layer and follow by flattened layers at the end. In looking at my model, it also contain repeating 3x3 convolution (follow by an Exponential Linear Unit activation) and flattened layer at the end before final activation. One important feature of this network is that the Input image will first go through a 1x1 convolution. The goal for this layer is to have the network determine which color channel is the most important when training the model. 

This model is using the Adam optimizer with the objective of reducing the mean squared error. The reason that Adam optimizer was chosen is because of the lower number of parameters to tune and the good performance. The model trained using 8 epochs, in this case, the number of epoch is important as a high value will overfit the model. 


Reflection
------------------------
This is a really interesting project in which it drive home the point of the important of size and variety of data. Image augmentation is extremely important for this project. Since we are only driving only in one track, it is important to not overfit the model to which it can only drive on that track. By randomly translating, fliping and changing the brightness of the image, I can basically create infinite set of training set from one image. This can in turn generalize the data set to where the car can drive in different condition. Sometime, having a good and large data set is better than having a good model. For follow up work, I believe futher tuning the hyperparameter, the car can drive smoother and less sudden jerk when doing aggressive turn.

Video of result (all training is done one track 1 and the model has not seen track 2)
------------------------
Track1: https://www.youtube.com/watch?v=hpTUgs1y-Eg
Track2: https://www.youtube.com/watch?v=LSm-btK1JJI



