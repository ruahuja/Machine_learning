#!usr/bin/python

import cv2, time,numpy

#reading image
img=cv2.imread("dog.jpeg")

#checking shape
print img.shape


#printing data
print img

#extracting only red color
img1=cv2.inRange(img,(0,0,0),(40,40,255))
cv2.imshow("onlyred",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
