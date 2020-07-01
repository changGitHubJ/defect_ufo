import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from PIL import Image

import common as c

ufo = ["./images/ufo.bmp"]
sky_files = ["./images/sky/01.JPG",
             "./images/sky/02.JPG",
             "./images/sky/03.JPG",
             "./images/sky/04.JPG",
             "./images/sky/05.JPG",
             "./images/sky/06.JPG",
             "./images/sky/07.JPG",
             "./images/sky/08.JPG",
             "./images/sky/09.JPG",
             "./images/sky/10.JPG"]
test = "./images/ufo/"

def deg2rad(val):
    return val*math.pi/180.0

if __name__ == "__main__":
    # # training data #############################
    ufo_org = Image.open(ufo[0])

    counter = 0
    train_img = []
    train_lbl = []
    for sky_file in sky_files:
        sky = Image.open(sky_file)
        sky = sky.convert("L")
        max_x = sky.size[0]
        max_y = sky.size[1]

        for n in range(c.DATA_SIZE):
            print("training data, counter = %d"%counter)

            # background
            size = random.randint(int(c.IMG_SIZE/2.0), int(c.IMG_SIZE*2)) #image size
            if(max_x > size):
                org_x = random.randint(0, max_x - size)
                size_x = size
            else:
                org_x = 0
                size_x = max_x
            if(max_y > size):
                org_y = random.randint(0, max_y - size)
                size_y = size
            else:
                org_y = 0
                size_y = max_y
            cropped512 = sky.crop([org_x, org_y, org_x + size_x, org_y + size_y])
            cropped512 = cropped512.resize([c.IMG_SIZE*2, c.IMG_SIZE*2])
            cropped512 = np.array(cropped512)

            # add ufo
            ufo = ufo_org.copy() # copy original image
            rate = random.uniform(0.2, 1.0) # size of 0.5 to 1.0 times            
            ufo = ufo.resize([int(ufo.size[0]*rate), int(ufo.size[1]*rate)])
            ufo_gray = np.array(ufo.convert("L"))# gray scale            
            ufo = np.array(ufo) # tp array
            bright = np.mean(cropped512)/np.mean(ufo)
            ufo = ufo*bright # adjust brightness

            x = random.randint(0, c.IMG_SIZE*2)
            y = random.randint(0, c.IMG_SIZE*2)
            rot = deg2rad(random.uniform(-30, 30)) # rotation angle 
            label512 = np.zeros([c.IMG_SIZE*2, c.IMG_SIZE*2])
            for i in range(ufo.shape[0]):
                for j in range(ufo.shape[1]):
                    if ufo[i, j, 0] != 255 and ufo[i, j, 1] != 0 and ufo[i, j, 2] != 2:
                        rot_i = int(i*math.cos(rot) - j*math.sin(rot) + 0.5)
                        rot_j = int(i*math.sin(rot) + j*math.cos(rot) + 0.5)
                        if 0 <= x + rot_i and x + rot_i < c.IMG_SIZE*2 and 0 <= y + rot_j and y + rot_j < c.IMG_SIZE*2:
                            cropped512[x + rot_i, y + rot_j] = ufo_gray[i, j]
                            label512[x + rot_i, y + rot_j] = 255
            image = np.zeros([c.IMG_SIZE, c.IMG_SIZE])
            label = np.zeros([c.IMG_SIZE, c.IMG_SIZE])
            for i in range(c.IMG_SIZE):
                for j in range(c.IMG_SIZE):
                    image[i, j] = (float(cropped512[2*i, 2*j]) + float(cropped512[2*i + 1, 2*j]) + float(cropped512[2*i, 2*j + 1]) + float(cropped512[2*i + 1, 2*j + 1]))/4.0
                    label[i, j] = (label512[2*i, 2*j] + label512[2*i + 1, 2*j] + label512[2*i, 2*j + 1] + label512[2*i + 1, 2*j + 1])/4.0

            # plot
            fig = plt.figure(figsize=[11, 4])
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap="gray", vmin=0, vmax=255)
            plt.tick_params(bottom=False, left=False, right=False, top=False,
                labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            plt.subplot(1, 2, 2)
            plt.imshow(label, cmap="gray", vmin=0, vmax=255)
            plt.tick_params(bottom=False, left=False, right=False, top=False,
                labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            plt.show()

            line_img = str(counter)
            line_lbl = str(counter)
            for i in range(c.IMG_SIZE):
                for j in range(c.IMG_SIZE):
                    line_img = line_img + ',' + str(image[i, j])
                    line_lbl = line_lbl + ',' + str(label[i, j])
            line_img = line_img + '\n'
            line_lbl = line_lbl + "\n"
            train_img.append(line_img)
            train_lbl.append(line_lbl)
            counter += 1

    if not os.path.exists('./data'):
        os.mkdir('./data')
    with open('./data/trainImage256.txt', 'w') as f:
        for line in train_img:
            f.write(line)
    with open('./data/trainLabel256.txt', 'w') as f:
        for line in train_lbl:
            f.write(line)
    
    counter = 0
    test_img = []
    test_lbl = []
    # test data #################################
    for n in range(c.TEST_DATA_SIZE):
        print("test data, counter = %d"%counter)
        image = Image.open(test + "%d_256.jpg"%n)
        label = Image.open(test + "%d_256_label.jpg"%n)
        image = image.convert("L")
        image = np.array(image)
        label = label.convert("L")
        label = np.array(label)

        # plot
        # fig = plt.figure(figsize=[11, 4])
        # plt.subplot(1, 2, 1)
        # plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        # plt.subplot(1, 2, 2)
        # plt.imshow(label, cmap="gray", vmin=0, vmax=255)
        # plt.show()

        line_img = str(counter)
        line_lbl = str(counter)
        for i in range(c.IMG_SIZE):
            for j in range(c.IMG_SIZE):
                line_img = line_img + ',' + str(image[i, j])
                line_lbl = line_lbl + ',' + str(label[i, j])
        line_img = line_img + '\n'
        line_lbl = line_lbl + "\n"
        test_img.append(line_img)
        test_lbl.append(line_lbl)
        counter += 1

    with open('./data/testImage256.txt', 'w') as f:
        for line in test_img:
            f.write(line)
    with open('./data/testLabel256.txt', 'w') as f:
        for line in test_lbl:
            f.write(line)
    