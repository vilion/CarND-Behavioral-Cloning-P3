# -*- coding:utf-8 -*-
import numpy as np
import cv2
import csv

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# generator's output the data feed to fit_generator 
correction = [0.0, 0.2, -0.2]
def generator(samples, batch_size=32):
    while 1:
        shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for line in batch_samples:
                for i in range(3):
                    sourcepath = line[i]
                    filename = sourcepath.split("/")[-1]
                    current_path = 'data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(line[3])+correction[i]
                    measurements.append(measurement)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(-1.0*measurement)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train,y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Cropping2D,Convolution2D
from keras.layers.core import Dropout

# model same as NVIDIA one
model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5)) # in test accuracy, dropout may ignored by framework
model.add(Dense(50))
model.add(Dropout(0.5)) # in test accuracy, dropout may ignored by framework
model.add(Dense(10))
model.add(Dense(1))

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

model.save('model.h5')
exit()
