# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:36:11 2022

@author: Administrator
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os 
file = "../dataset"
for dataset in os.listdir(file):
    for x in os.listdir(os.path.join(file, dataset)):
        if x == "generator_0":
            for i in os.listdir(os.path.join(file, dataset, x)):
                img = cv2.imread(os.path.join(file, dataset, x, i),
                                 1)  # 0是第二个参数，将其转为灰度图
                n = 45
                m = 255
                ret, thresh1 = cv2.threshold(img, n, m, cv2.THRESH_BINARY)
                if not os.path.exists(os.path.join(file, dataset, "thresh_0")):
                    os.mkdir(os.path.join(file, dataset, "thresh_0"))
                cv2.imwrite(os.path.join(file, dataset, "thresh_0", i), thresh1)
        elif x == "generator_90":
            for i in os.listdir(os.path.join(file, dataset, x)):
                img = cv2.imread(os.path.join(file, dataset, x, i),
                                 1)  # 0是第二个参数，将其转为灰度图
                n = 45
                m = 255
                ret, thresh1 = cv2.threshold(img, n, m, cv2.THRESH_BINARY)
                if not os.path.exists(os.path.join(file, dataset, "thresh_90")):
                    os.mkdir(os.path.join(file, dataset, "thresh_90"))
                cv2.imwrite(os.path.join(file, dataset, "thresh_90", i), thresh1)
