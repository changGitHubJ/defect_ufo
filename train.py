import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import time

from PIL import Image

import load_data as data
import model

import common as c

# Parameter
training_epochs = 200
batch_size = 8

def main(data, model):
    print("Reading images...")
    x_train = data.read_images('./data/trainImage256.txt', c.TRAIN_DATA_SIZE)
    x_test = data.read_images('./data/testImage256.txt', c.TEST_DATA_SIZE)
    y_train = data.read_images('./data/trainLabel256.txt', c.TRAIN_DATA_SIZE)
    y_test = data.read_images('./data/testLabel256.txt', c.TEST_DATA_SIZE)
    
    print("Creating model...")
    model.create_model()

    print("Now training...")
    history = model.training(x_train, y_train, x_test, y_test)
    accuracy = history.history["accuracy"]
    loss = history.history["loss"]
    eval = model.evaluate(x_test, y_test)
    
    print("accuracy = " + str(eval))
    model.save('./model.h5')

    ax1 = plt.subplot()
    ax1.plot(loss, color="blue")
    ax2 = ax1.twinx()
    ax2.plot(accuracy, color="orange")
    plt.show()

if __name__=='__main__':
    data = data.MyLoadData(c.IMG_SIZE, c.OUTPUT_SIZE)
    model = model.MyModel((c.IMG_SIZE, c.IMG_SIZE, 1), batch_size, training_epochs)
    main(data, model)