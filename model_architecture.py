
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Lambda, Convolution2D, MaxPooling2D, Flatten,Dropout,Reshape
#whether to use dropout:
drop=True

def this_model(input_shape):
	#define the architecture of your model here:
	if drop:
		model = Sequential()
		model.add(Dense(300,input_shape = input_shape))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(100))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(10))
		model.add(Activation('softmax'))
	else:
		model = Sequential()
		model.add(Dense(300,input_shape = input_shape))
		model.add(Activation('relu'))
		#model.add(Dropout(0.5))
		model.add(Dense(100))
		model.add(Activation('relu'))
		#model.add(Dropout(0.5))
		model.add(Dense(10))
		model.add(Activation('softmax'))
	return model