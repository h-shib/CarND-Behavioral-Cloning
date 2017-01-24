import os
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Input, Flatten, Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

image_file_path = '/Users/hiraku/Desktop/IMG/'
driving_file_path = '/Users/hiraku/Desktop/driving_log.csv'
images = os.listdir(image_file_path)

def load_data():
	train_imgs = []
	train_imgs = np.array([mpimg.imread(image_file_path + img) for img in images])
	print(train_imgs)

	train_labels = pd.read_csv(driving_file_path, header=None).iloc[:, 3]

	return train_test_split(train_imgs, train_labels, test_size=0.3, random_state=0)

def main():
	X_train, X_test, y_train, y_test = load_data()

	print("Shape of Train: ", X_train.shape, y_train.shape)
	print("Shape of Test: ", X_test.shape, y_test.shape)

	input_shape = X_train.shape[1:]
	output_shape = len(np.unique(y_train))

	print(input_shape)
	print(output_shape)

	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(160,320,3,)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))

	#inputs = Input(shape=input_shape)
	#x = Flatten()(inputs)
	#prediction = Dense(output_shape, activation='softmax')(x)

	#model = Model(input=inputs, output=prediction)
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	model.fit(X_train, y_train, shuffle=True, validation_data=(X_test, y_test), nb_epoch=5)

	print(model.get_config)
	with open('model.json', 'w') as f:
		f.write(model.to_json())
	model.save_weights('model.h5')

if __name__ == '__main__':
	main()
	
    
