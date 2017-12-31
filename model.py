import csv
import cv2
import numpy as np
import os
import random
from PIL import Image


class DataGenerator(object):
  """DataGenerator class generates the training and validation batches used in Keras model training.
  """

  def __init__(self, image_paths, measurements, shuffle=True):
    """The data set is automatically splitted into training and validation sets by 80:20 ratio.
    Args:
      image_paths: a list of paths to the training images.

      measurements: a list of steering angles corresponding to each of the training image.

    Note:
      Each input image is augmented by flipping it left to right.
    """
    self._image_paths = image_paths
    self._measurements = measurements
    assert(len(self._image_paths) == len(self._measurements))
    self._done_one_round = False
    self._shuffle = shuffle
    self._batch_size = 64
    self._validation_ratio = 0.2
    n = int((1 - self._validation_ratio) * len(self._measurements))
    self._train_image_paths, self._valid_image_paths = self._image_paths[:n], self._image_paths[n:]
    self._train_measurements, self._valid_measurements = self._measurements[:n], self._measurements[n:]
    self._num_train = n
    self._num_valid = len(self._measurements) - n
    self._image_shape = cv2.imread(image_paths[0]).shape

  def generate(self, mode='train'):
    """Generate a generator based on the mode.

    Args:
      mode: a str, either 'train' or 'valid'.
    """
    if mode == 'train':
      mode_image_paths, mode_measurements = self._train_image_paths, self._train_measurements
    else:
      mode_image_paths, mode_measurements = self._valid_image_paths, self._valid_measurements
    while True:
      if self._shuffle:
        temp = list(zip(mode_image_paths, mode_measurements))
        random.shuffle(temp)
        image_paths, measurements = zip(*temp)
      else:
        image_paths, measurements = mode_image_paths, mode_measurements
      n = len(measurements)
      for i in range(0, n - self._batch_size, self._batch_size):
        X, y = [], []
        for j in range(i, min(n, (i + self._batch_size))):
          image_path, measurement = image_paths[j], measurements[j]
          image = np.array(Image.open(image_path).getdata()).reshape(self._image_shape)
          X.append(image)
          y.append(measurement)
          X.append(np.fliplr(image))
          y.append(-measurement)
        yield np.array(X), np.array(y)

  def nb_train_samples(self):
    """Number of training samples.
    """
    return (self._num_train // self._batch_size) *  self._batch_size * 2  # data augmentation factor of 2

  def nb_valid_samples(self):
    """Number of validation samples.
    """
    return (self._num_valid // self._batch_size) *  self._batch_size * 2  # data augmentation factor of 2

  def image_shape(self):
    """Shape of the image.
    """
    return self._image_shape


# Create a list of image paths and corresponding steering angles. And use them to initialize a DataGenerator.
data_dir = 'data'
image_paths, measurements = [], []
steer_correction = [0.0, 0.2, -0.2]
with open(os.path.join(data_dir, 'driving_log.csv'), 'r') as f:
  reader = csv.reader(f)
  header = True
  for line in reader:
    if header:
      header = False
      continue
    for i in range(3):
      image_paths.append(os.path.join(data_dir, line[i].strip()))
      measurements.append(float(line[3]) + steer_correction[i])
dgen = DataGenerator(image_paths, measurements)

# Create the training and validation data generators.
train_data = dgen.generate(mode='train')
valid_data = dgen.generate(mode='valid')

# Construct a Keras model.
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Convolution2D, MaxPooling2D, Dropout, Activation, Lambda, AveragePooling2D

model = Sequential()

# Preprocessing.
model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=dgen.image_shape()))
model.add(Cropping2D(cropping=((70, 20), (0, 0))))
model.add(AveragePooling2D())

# VGG-like network.
model.add(Convolution2D(32, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# Train the model.
model.compile(loss='mse', optimizer='adam')
model.fit_generator(generator=train_data, samples_per_epoch=dgen.nb_train_samples(), nb_epoch=4,
                    validation_data=valid_data, nb_val_samples=dgen.nb_valid_samples())

# Save the model.
model.save('model.h5')
