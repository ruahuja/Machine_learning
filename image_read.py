#!/usr/bin/python
import numpy
import cv2

#loding image
img=cv2.imread("dog.jpeg")
img1=cv2.imread("dog1.jpeg")

#print height abd width
#print (img.shape)

#print to display image
cv2.imshow('dog',img)
cv2.imshow('dog1',img1)

#image window holder activate
cv2.waitKey(0)
cv2.destroyAllWindows()




