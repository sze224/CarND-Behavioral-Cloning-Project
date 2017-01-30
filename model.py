# load library/ function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
from sklearn.utils import shuffle
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Dropout, merge, Lambda
import pickle

# define functions
# min_max_scaling function to normalize images

def get_batch(X, y, keep_prob, batch_size):
    X_batch = np.zeros((batch_size, 35, 140, 3))
    y_batch = np.zeros((batch_size))
    while 1:
        for i in range (batch_size):
            done = 0
            while done == 0:
                ind = np.random.randint(0, len(y)-1)
                pos = np.random.randint(0,2)
                img = cv2.imread(X[pos][ind].strip())
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if pos == 0:
                    steer = y[ind]
                elif pos == 1:
                    steer = y[ind] + .18
                elif pos == 2:
                    steer = y[ind] - .18
                X_1, y_1 = image_augmentation(img, steer)
                if abs(y_1) < 0.1:
                    drop = np.random.uniform()
                    if drop > keep_prob:
                        done = 1
                else:
                    done = 1
            X_1 = X_1[60:130:2, 20:300:2, :]
            X_batch[i] = X_1
            y_batch[i] = y_1    
        yield X_batch, y_batch


def image_augmentation(img, steer_ang):
    
    steer_bias = 0.002
    # randomly translate image to mimic car at different position
    t_x = 100*np.random.uniform()-100/2
    t_y = 10*np.random.uniform()-10/2
    trans = np.float32([[1,0,t_x],[0,1,t_y]])
    trans_img = cv2.warpAffine(img,trans,(img.shape[1],img.shape[0]))
    new_steer = steer_ang + t_x * steer_bias

    # randomly change brightness to mimic different time of day
    light_bias = .25 + np.random.uniform()
    img_out = cv2.cvtColor(trans_img, cv2.COLOR_RGB2HSV)
    img_out[:,:,2] = img_out[:,:,2] * light_bias
    img_out = cv2.cvtColor(img_out, cv2.COLOR_HSV2RGB)
    
    # flip image 50 percent of the time
    if np.random.randint(0,2) == 1:
        img_out = cv2.flip(img_out,1)
        new_steer = -new_steer
    
    return img_out, new_steer

# read in image names and steer angle from csv file
data = pd.read_csv('driving_log.csv', header = None)
C = data[0]
L = data[1]
R = data[2]
steer_angle = data[3]
X = [C, L, R]

# Model Structure 
input_shape = (35, 140, 3)
inp = Input(shape = input_shape)

# first layer: 1x1 convolution to pick out the best color channel
x1 = Lambda(lambda x: x/255. -0.5, input_shape=input_shape)(inp)

# second layer: 2 3X3 convolution 
x2 = Conv2D(32,3,3, border_mode='same', activation = 'relu')(x1)
x2 = Conv2D(32,3,3, border_mode='same', activation = 'relu')(x2)
x2 = MaxPooling2D((2,2))(x2)
x2 = Dropout(0.5)(x2)

# third layer: 2 3X3 convolution
x3 = Conv2D(64,3,3, border_mode='same', activation = 'relu')(x2)
x3 = Conv2D(64,3,3, border_mode='same', activation = 'relu')(x3)
x3 = MaxPooling2D((2,2))(x3)
x3 = Dropout(0.5)(x3)

# third layer: 2 3X3 convolution
x4 = Conv2D(64,3,3, border_mode='same', activation = 'relu')(x3)
x4 = Conv2D(64,3,3, border_mode='same', activation = 'relu')(x4)
x4 = MaxPooling2D((2,2))(x4)
x4 = Dropout(0.5)(x4)

x5 = Flatten()(x4)
x6 = Dense(512, activation = 'relu')(x5)
x7 = Dense(128, activation = 'relu')(x6)
out = Dense(1, activation = 'linear')(x7)


model = Model(inp, out)
model.summary()

# compile model using adam optimizer and mean squared error
model.compile(optimizer = 'adam', loss = 'mse')

# train model 
print('Training...')
for j in range (5):
    keep_prob = 1/(j+1)  
    model.fit_generator(get_batch(X, steer_angle, keep_prob, 256), nb_epoch = 1, samples_per_epoch = 20000)

# save model and weight
model_json = model.to_json()
with open("model.json", "w") as f:
    f.write(model_json)

model.save_weights('model.h5')
print('Saved Model and weights')
