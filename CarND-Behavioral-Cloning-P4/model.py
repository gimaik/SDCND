import numpy as np
import pandas as pd
import math
import os
import csv
import cv2

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def flip(image, angle):	
	image = cv2.flip(image, 1)
	angle = -angle
	return image, angle

def random_brightness(image, angle):
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	ratio =0.25 + np.random.rand()
	hsv[:,:,2] =  hsv[:,:,2] * ratio
	image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	return image, angle

def random_translate(image, angle, range_x, range_y):
	trans_x = range_x * (np.random.rand() - 0.50)
	trans_y = range_y * (np.random.rand() - 0.50)
	angle += trans_x * 0.002
	trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
	height, width = image.shape[:2]
	image = cv2.warpAffine(image, trans_m, (width, height))
	return image, angle


# Create a generator function for training the network
def generator(samples, batch_size=32):
	'''Generate image input features and steering angle target
	Camera positions: center 0, left 1, right 2
	'''
	num_samples = len(samples)
	CORRECTION = [0.00, 0.20, -0.20]
	
	while 1:  # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset + batch_size]
			images, angles = [], []

			for batch_sample in batch_samples:

				for i in range(3):
					filename = './data/' + batch_sample[i]
					image = cv2.imread(filename)
                   
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					angle = batch_sample[3] + CORRECTION[i]                    
					images.append(image)
					angles.append(angle)
                    
					image_tr, angle_tr = random_translate(image, angle, 50, 20)
					images.append(image_tr)
					angles.append(angle_tr)       
                    
					image_fl, angle_fl = flip(image, angle) 
					images.append(image_fl)
					angles.append(angle_fl)
                    
			X_train = np.array(images)
			y_train = np.array(angles)
			yield shuffle(X_train, y_train)

def train_model(batch_size = 32):
	HEIGHT, WIDTH, CHANNEL = 160, 320, 3
	CROP_TOP, CROP_BOTTOM = 70, 25
	EPOCH = 10

	'''Create the network, train, and return.'''	
	samples = pd.read_csv('data/driving_log.csv').values

	# Set up training & validation sets
	train_samples, valid_samples = train_test_split(samples, test_size=0.25)	
	train_generator = generator(train_samples, batch_size=batch_size)
	valid_generator = generator(valid_samples, batch_size=batch_size)

	# Create a model
	model = Sequential()	
	model.add(Lambda(lambda x: x/255.0 - 0.50, input_shape=(HEIGHT, WIDTH, CHANNEL)))
	model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0, 0))))    	
	model.add(Conv2D(36, (3, 3), strides=(2, 2), activation='elu'))
	model.add(Conv2D(48, (3, 3), strides=(2, 2), activation='elu'))
	model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='elu'))
	model.add(Conv2D(128, (3, 3), activation='elu'))
	model.add(Conv2D(128, (3, 3), activation='elu'))
	model.add(MaxPooling2D())
	model.add(Flatten())

	model.add(Dense(256, activation='elu'))
	model.add(Dropout(0.50))
	model.add(Dense(128, activation='elu'))
	model.add(Dropout(0.50))
	model.add(Dense(10, activation='elu'))	
	model.add(Dense(1))

	# Train the model	
	model.compile(loss='mse', optimizer='adam')

	early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
	checkpoint = ModelCheckpoint("./saved_models/model-{epoch:03d}.h5", monitor="val_loss", 
								verbose=0, save_best_only=True)
		
	model.fit_generator(train_generator, 
						steps_per_epoch = int(len(train_samples) / batch_size), 
						validation_data=valid_generator, 
						validation_steps = int(len(valid_samples) / batch_size),
						callbacks=[early_stopping, checkpoint], 
						nb_epoch=EPOCH)
	model.save('model.h5')

	return model


if __name__ == '__main__':
	model = train_model()
	print('Model saved.')
