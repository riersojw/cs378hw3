import cv2
import numpy as np

img = cv2.imread('RedCircle.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
print(contours[0])
M = cv2.moments(cnt)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

cX2 = (cX + 10)
cY2 = (cY + 10)
ecX2 = (cX -10)
ecY2 = (cY -10)
print(M)
print(cX)
print(cY)
cv2.line(img, ((cX - 10), (cY - 10)), ((cX + 10), (cY + 10)), (0, 255, 0), 2)
cv2.line(img, ((cX + 10), (cY - 10)), ((cX - 10), (cY + 10)), (0, 255, 0), 2)

x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

