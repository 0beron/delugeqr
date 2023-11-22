import cv2 as cv

import numpy as np

nx, ny = (14, 6)
x = np.linspace(0, 13, nx)
y = np.linspace(0, 5, ny)

mg = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)
print(mg)
mg2 = np.concatenate((mg, np.array([[55.0,55.0]])), axis=0)

h, status = cv.findHomography(mg, mg2, cv.RANSAC)


print(h)
