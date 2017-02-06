import os
import cv2
import csv
import math
import numpy as np
from scipy.misc import imresize
import matplotlib.image as mpimg
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

image_file_path = './data/IMG/'
driving_file_path = './data/driving_log.csv'

conv_layers = [32, 32, 64, 128]
dense_layers = [1024, 512, 256]

def model(input_shape):
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, activation='elu', input_shape=input_shape))
	model.add(MaxPooling2D())
	for layers in conv_layers:
		model.add(Convolution2D(layers, 3, 3, activation='elu'))
		model.add(MaxPooling2D())
	model.add(Flatten())
	for layers in dense_layers:
		model.add(Dense(layers, activation='elu'))
		model.add(Dropout(0.5))
	model.add(Dense(1, activation='linear'))
	model.compile(loss='mse', optimizer='adam')
	return model

def load_data():
	X, y = [], []
	offset = 0.4
	with open(driving_file_path) as f:
		reader = csv.reader(f)
		next(reader)
		for center, left, right, steering, _, _, _ in reader:
			X += [left.strip(), right.strip()]
			y += [float(steering)+offset, float(steering)-offset]
	return X, y

def preprocess_image(image_path, steering, shape=(100, 100)):
	image = mpimg.imread(image_path)
	origin_shape = image.shape
	image = image[math.floor(float(origin_shape[0])/5):origin_shape[0]-25, 0:origin_shape[1]]
	image = imresize(mpimg.imread(image_path), shape)
	image = random_brightness(image)
	image, steering = random_flip(image, steering)
	image = (image / 255. - 0.5)
	return image, steering

def random_flip(image, steering):
	is_flip = np.random.randint(2)
	if is_flip:
		image = cv2.flip(image, 1)
		steering = -steering
	return image, steering

def random_brightness(image):
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	random_brightness = np.random.uniform() + 0.25
	image[:, :, 2] = image[:, :, 2] * random_brightness
	image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
	return image

def _generator(batch_size, X, y):
	while 1:
		batch_X, batch_y = [], []
		for i in range(batch_size):
			random_idx = np.random.randint(len(X))
			steering = y[random_idx]
			image_path = "./data/" + X[random_idx]
			image, steering = preprocess_image(image_path, steering)
			batch_X.append(image)
			batch_y.append(steering)
		yield np.array(batch_X), np.array(batch_y)

def main():
	X, y = load_data()
	net = model(input_shape=(100, 100, 3))
	net.fit_generator(_generator(256, X, y), samples_per_epoch=20224, nb_epoch=8)
	with open('checkpoints/model.json', 'w') as f:
		f.write(net.to_json())
	net.save_weights('./checkpoints/model.h5')
	print("model saved")

if __name__ == '__main__':
	main()