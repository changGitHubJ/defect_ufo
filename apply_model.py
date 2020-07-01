
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from matplotlib import cm
from PIL import Image

import common as c
import load_data as data

if __name__=='__main__':    
    with tf.device("/gpu:0"):
        model = keras.models.load_model("model.h5")
        directory = "./images/ufo"
        files = os.listdir(directory)
        for file in files:
            if "_256.jpg" in file:
                image_org = Image.open(directory + "/" + file)
                image_gray = image_org.convert("L")
                image = np.array(image_gray)
                infered = model.predict(image.reshape([1, c.IMG_SIZE, c.IMG_SIZE, 1]))

                infered = infered.reshape([c.IMG_SIZE, c.IMG_SIZE])
                sum_col = infered.sum(axis=0)
                area_col = []
                zero_thres = np.max(sum_col)*0.5
                start_col = -1
                for m in range(c.IMG_SIZE - 1):
                    if(sum_col[m] < zero_thres and sum_col[m + 1] >= zero_thres):
                        start_col = m
                    if(sum_col[m] >= zero_thres and sum_col[m + 1] < zero_thres):
                        if(start_col > 0 and start_col != m):
                            area_col.append([start_col, m])
                area = []
                for k in range(len(area_col)):
                    start_col = area_col[k][0]
                    end_col = area_col[k][1]
                    temp = infered[:, start_col:end_col]
                    sum_row = temp.sum(axis=1)
                    zero_thres = np.max(sum_row)*0.5
                    start_row = -1
                    for n in range(c.IMG_SIZE - 1):
                        if(sum_row[n] < zero_thres and sum_row[n + 1] >= zero_thres):
                            start_row = n
                        if(sum_row[n] >= zero_thres and sum_row[n + 1] < zero_thres):
                            if(start_row > 0 and start_row != n):
                                area.append([start_col, end_col, start_row, n])
                
                result = np.array(image_org.copy())
                for k in range(len(area)):
                    s_col = area[k][0]
                    e_col = area[k][1]
                    s_row = area[k][2]
                    e_row = area[k][3]
                    for i in range(s_col,e_col):
                        result[s_row, i, 0] = 255
                        result[s_row, i, 1] = 0
                        result[s_row, i, 2] = 0
                        result[e_row, i, 0] = 255
                        result[e_row, i, 1] = 0
                        result[e_row, i, 2] = 0
                    for j in range(s_row, e_row):
                        result[j, s_col, 0] = 255
                        result[j, s_col, 1] = 0
                        result[j, s_col, 2] = 0
                        result[j, e_col, 0] = 255
                        result[j, e_col, 1] = 0
                        result[j, e_col, 2] = 0
                
                plt.figure(figsize=[11, 4])
                plt.subplot(1, 3, 1)
                fig = plt.imshow(image_org)
                plt.tick_params(bottom=False, left=False, right=False, top=False,
                                labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                plt.subplot(1, 3, 2)
                plt.imshow(infered, cmap="gray", vmin=0.0, vmax=1.0)
                plt.tick_params(bottom=False, left=False, right=False, top=False,
                                labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                plt.subplot(1, 3, 3)
                plt.imshow(result)
                plt.tick_params(bottom=False, left=False, right=False, top=False,
                                labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                plt.show()