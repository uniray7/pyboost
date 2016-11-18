import cv2
import numpy as np
import pbcvt


img = cv2.imread('test.jpg')
print(type(img))
gray_img =  pbcvt.grayscale(img)
print(type(gray_img))
cv2.imwrite('gray.jpg', gray_img)
