Defecting: Only one image with defection
====

Overview

- Deep learning for defecting
- The number of images with defection is very small
- U-net algorithm
- keras + tensorflow

## Requirement

- python 3.6.9
- tensorflow-gpu 1.14.0
- keras 2.3.1

## Usage

1. Configurate image and lable  
Start configurate_data.py, then the followings are created.

- ./data/trainImage256.txt
- ./data/trainLabel256.txt
- ./data/testImage256.txt
- ./data/testLabel256.txt

You can check the data using check_data.py

1. Training with tensorflow  
Start train.py

1. Check results  
Please store ufo image files to ./images/ufo, then start ./images/resize_image.py.  
Start apply_model.py

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)