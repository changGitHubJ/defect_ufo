import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image

import common as c
NUM = 30

def readImages(filename):
    images = np.zeros((NUM, c.IMG_SIZE, c.IMG_SIZE, 1))
    fileImg = open(filename)
    for k in range(NUM):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(c.IMG_SIZE):
            for j in range(c.IMG_SIZE):
                images[k, i, j, 0] = float(val[c.IMG_SIZE*i + j + 1])
    return images

if __name__=='__main__':
    train_image = readImages('./data/trainImage256.txt')
    train_label = readImages('./data/trainLabel256.txt')

    for i in range(NUM):
        plt.figure(figsize=[10, 4])
        plt.subplot(1, 2, 1)
        fig = plt.imshow(train_image[i, :].reshape([c.IMG_SIZE, c.IMG_SIZE]))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)    
        
        plt.subplot(1, 2, 2)
        fig = plt.imshow(train_label[i, :].reshape([c.IMG_SIZE, c.IMG_SIZE]))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.show()