import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def empty(a):
	pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 33, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 135, 255, empty)

while (cap.isOpened()):
	ret, frame=cap.read()
	frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        bframe = cv2.GaussianBlur(frame, (7, 7), 1)
	fgray = cv2.cvtColor(bframe, cv2.COLOR_BGR2GRAY)

        lred = np.array([105, 150, 50])
        ured = np.array([125, 255, 255])
        mask = cv2.inRange(hsv, lred, ured)

        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        imgCanny = cv2.Canny(fgray, threshold1, threshold2)

        contours, hierarchy = cv2.findContours(mask, 1, 2)
        ret, thresh = cv2.threshold(fgray, 127, 255, 0)
        gcontours, ghierarchy = cv2.findContours(thresh, 1, 2)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

	cv2.imshow('Frame', frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Canny", imgCanny)
        cv2.imshow("Gray", fgray)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()

cv2.destroyAllWindows()

