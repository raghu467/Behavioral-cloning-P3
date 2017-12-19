import errno
import json
import os
import cv2

from scipy.ndimage import rotate
from scipy.stats import bernoulli

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc

# Data Prep Constants
DRIVING_LOG_FILE = './data/driving_log.csv'
IMG_PATH = './data/'
STEERING_CONSTANT = 0.229


def crop_image(image, top_percent, bottom_percent):
    # Check if the entire image should not be cropped
    assert 0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    assert 0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]


def resize_image(image, new_dim):
    return scipy.misc.imresize(image, new_dim)


def flip_image(image, steering_angle, flipping_prob=0.6):
    # if image is flipped the steering angle needs to be negated
    coin = bernoulli.rvs(flipping_prob)
    #print("flip coin =" ,coin)
    # if head is true then flip the image and negate the steering angle and return
    if coin == 0:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


def random_gamma(image):
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def random_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4 * (2 * np.random.uniform() - 1.0)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def random_shear(image, steering_angle, shear_range=200 ):
    # Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
        rows, cols, ch = image.shape
        dx = np.random.randint(-shear_range, shear_range + 1)
        random_point = [cols / 2 + dx, rows / 2]
        pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
        pts2 = np.float32([[0, rows], [cols, rows], random_point])
        dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
        steering_angle += dsteering
        return image, steering_angle


def preporcess_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                     resize_size=(64, 64), shear_prob=0.9):


    # Image pre-processing Pipeline
    image, steering_angle = random_shear(image, steering_angle, shear_prob)

    image = crop_image(image, top_crop_percent, bottom_crop_percent)

    image, steering_angle = flip_image(image, steering_angle)

    image = random_brightness(image)

    image = resize_image(image, resize_size)

    return image, steering_angle
def read_next_image_files_angles(batch_size=64):

    # Randomlly select the images(center left or right) and also select the corresponding steering angle.
    # also adjust the steering angle using the steering_coefficient for the left and right images

    training_data = pd.read_csv(DRIVING_LOG_FILE)
    num_of_img = len(training_data)
    random_index = np.random.randint(0, num_of_img, batch_size)
    image_files_and_angles = []
    for index in random_index:
        random_image = np.random.randint(0, 3)
        if random_image == 0:      #left
            img = training_data.iloc[index]['left'].strip()
            steering_angle = training_data.iloc[index]['steering'] + STEERING_CONSTANT
            image_files_and_angles.append((img, steering_angle))

        elif random_image == 1:    #center
            img = training_data.iloc[index]['center'].strip()
            steering_angle = training_data.iloc[index]['steering']
            image_files_and_angles.append((img, steering_angle))
        else:                        #right
            img = training_data.iloc[index]['right'].strip()
            steering_angle = training_data.iloc[index]['steering'] - STEERING_CONSTANT
            image_files_and_angles.append((img, steering_angle))

    return image_files_and_angles

def generate_next_batch(batch_size=64):

    while True:
        X_batch = []
        y_batch = []
        images_and_angles = read_next_image_files_angles(batch_size)
        for img_file, angle in images_and_angles:
            raw_image = plt.imread(IMG_PATH+img_file)
            raw_angle = angle
            new_image, new_angle = preporcess_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be 64'
        yield np.array(X_batch), np.array(y_batch)
def save_model(model, model_name='model.json', weights_name='model.h5'):

    delete_file(model_name)
    delete_file(weights_name)

    json_string = model.to_json()
    with open(model_name, 'w') as outfile:
        json.dump(json_string, outfile) # save the model
    model.save("my_model.h5")
    model.save_weights(weights_name) # save the weights

def delete_file(file):

    try:
        os.remove(file)

    except OSError as error:
        if error.errno != errno.ENOENT:
            raise
