import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

import data_prep

tf.python.control_flow_ops = tf

epochs = 8
samples_per_epoch = 20032
validation_samples = 6400
learning_rate = 0.0001
activation_relu = 'relu'

#Reference:
# "End to End Learning for Self-Driving Cars" paper by NVIDIA'S
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

model = Sequential()

# Normalization
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

# starts with five convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# fully connected layers
model.add(Dense(1164))
model.add(Activation(activation_relu))

model.add(Dense(100))
model.add(Activation(activation_relu))

model.add(Dense(50))
model.add(Activation(activation_relu))

model.add(Dense(10))
model.add(Activation(activation_relu))

model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate), loss="mse", )

#generators for training and validation
train_generators = data_prep.generate_next_batch()
validation_generators = data_prep.generate_next_batch()



history = model.fit_generator(train_generators,
                              samples_per_epoch=samples_per_epoch,
                              nb_epoch=epochs,
                              validation_data=validation_generators,
                              nb_val_samples=validation_samples,
                              verbose=1)

# save the mode.kson(model) and mode.h5(weights)
data_prep.save_model(model)
