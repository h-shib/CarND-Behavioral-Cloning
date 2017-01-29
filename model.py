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
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from scipy.misc import imresize

image_file_path = './data/IMG/'
driving_file_path = './data/driving_log.csv'
images = os.listdir(image_file_path)

def load_data():
	train_imgs = [imresize(mpimg.imread(image_file_path + img)[60:130], (66, 200, 3)) for img in images]
	train_imgs = np.array(train_imgs)
	print("finish train_imgs")
	train_labels = pd.read_csv(driving_file_path).iloc[:, 3]
	train_labels = np.concatenate((train_labels, (train_labels+.2), (train_labels-.2)), axis=0)

	return train_test_split(train_imgs, train_labels, test_size=0.3, random_state=0)

def main():
	X_train, X_test, y_train, y_test = load_data()

	print("Shape of Train: ", X_train.shape, y_train.shape)
	print("Shape of Test: ", X_test.shape, y_test.shape)

	input_shape = X_train.shape[1:]
	output_shape = len(np.unique(y_train))

	model = Sequential()
	# Input Layer.
	# Input: 66x200x3. Output: normalized input
	model.add(Lambda(lambda x : x/127.5 - 1., input_shape = (66, 200, 3)))
	# Layer 1: Convolutional Layer.
	# Input: 66x200x3. Output: 66x200x24.
	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
	model.add(ELU())
	# Layer 2: Convolutional Layer.
	# Input: 66x200x24. Output: 66x200x36.
	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
	model.add(ELU())
	# Layer 3: Convolutional Layer.
	# Input: 66x200x36. Output: 66x200x48.
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
	model.add(ELU())
	# Layer 4: Convolutional Layer.
	# Input: 66x200x48. Output: 66x200x64.
	model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init='he_normal'))
	model.add(ELU())
	# Layer 5: Convolutional Layer.
	# Input: 66x200x64. Ouput: 66x200x64.
	model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init='he_normal'))
	model.add(ELU())
	model.add(Flatten())
	# Layer 6: Fully Connected Layer.
	# Input: 844800. Output: 1164.
	model.add(Dense(1164, init='he_normal'))
	model.add(ELU())
	# Layer 7: Fully Connected Layer.
	# Input: 1164. Output: 100.
	model.add(Dense(100, init='he_normal'))
	model.add(ELU())
	# Layer 8: Fully Connected Layer.
	# Input: 100. Output: 100.
	model.add(Dense(100, init='he_normal'))
	model.add(ELU())
	# Layer 9: Fully Connected Layer. (with dropout rate of 0.4)
	# Input: 100. Output: 10.
	model.add(Dropout(0.4))
	model.add(Dense(10, init='he_normal'))
	model.add(ELU())
	# Output Layer.
	model.add(Dense(1, init='he_normal'))

	#inputs = Input(shape=input_shape)
	#x = Flatten()(inputs)
	#prediction = Dense(output_shape, activation='softmax')(x)

	#model = Model(input=inputs, output=prediction)
	model.compile(loss='mean_squared_error', optimizer='adam')

	model.fit(X_train, y_train, shuffle=True, validation_data=(X_test, y_test), nb_epoch=1)

	with open('model.json', 'w') as f:
		f.write(model.to_json())
	model.save_weights('model.h5')

if __name__ == '__main__':
	main()
