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

        lred = np.array([0, 50, 50])
        ured = np.array([10, 255, 255])
        mask0 = cv2.inRange(hsv, lred, ured)

        lred = np.array([170, 50, 50])
        ured = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lred, ured)

        mask = mask0 + mask1

        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        imgCanny = cv2.Canny(fgray, threshold1, threshold2)
	kernel = np.ones((5, 5))
	imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(mask, 1, 2)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

	cv2.imshow('Frame', frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Canny", imgCanny)
        cv2.imshow("imgDil", imgDil)
        #cv2.imshow("Gray", fgray)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()

cv2.destroyAllWindows()

