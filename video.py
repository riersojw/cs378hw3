import cv2
import numpy as np

# get video from web cam
cap = cv2.VideoCapture(0)

# place function to pass emtpy
def empty(a):
	pass

# Trackbar to help smooth out the contours
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 33, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 135, 255, empty)

#function to find the contours of the object we are tracking
def getContours(img, imgContour):

	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
		# find the area of the contour
		area = cv2.contourArea(cnt)
		# only track objects larger than 1000; this reduces the noise
		if area > 1000:

			# Get the perimeter length
			peri = cv2.arcLength(cnt, True)
			# Approximate a polygonal Curve based on perimeter length
			approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
			# with approx now we can find the area to bound in a tracking box
			x, y, w, h = cv2.boundingRect(approx)	
			cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

			# find the center points of the object
			M = cv2.moments(cnt)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])

			# draw an x in the center
			cv2.line(imgContour, ((cX - 10), (cY - 10)), ((cX + 10), (cY + 10)), (0, 255, 0), 2)
			cv2.line(imgContour, ((cX + 10), (cY - 10)), ((cX - 10), (cY + 10)), (0, 255, 0), 2)

while (cap.isOpened()):
        # read in the video frame by frame
	ret, frame=cap.read()
        # make the video feed smaller
	frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
        # copy the video frame to manipulate 
        imgContour = frame.copy()

	# Blur the image This will reduce noise
        bframe = cv2.GaussianBlur(frame, (7, 7), 1)
	# Convert the image to Gray scale
	fgray = cv2.cvtColor(bframe, cv2.COLOR_BGR2GRAY)
	# Convert the image to Hue, Saturation, Value - So we can track the color red
	hsv = cv2.cvtColor(bframe, cv2.COLOR_BGR2HSV)

	# NP array lower limit of red and upper limit of red adding the two mask help reduce other colors
	# being white it tracks my red neckness self pretty good
        lred = np.array([0, 50, 50])
        ured = np.array([10, 255, 255])
        mask0 = cv2.inRange(hsv, lred, ured)

        lred = np.array([170, 50, 50])
        ured = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lred, ured)

        mask = mask0 + mask1

	#get threshold values for the canny function
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        imgCanny = cv2.Canny(fgray, threshold1, threshold2)
	# dilate the image further reducing noise
	kernel = np.ones((5, 5))
	imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

	#getContours and draw the X and put a box around the image
        getContours(mask, imgContour)

	#cv2.imshow('Frame', frame)
        # display the Mask feed showing color red is only being detected
        cv2.imshow("Mask", mask)
        #cv2.imshow("Canny", imgCanny)
        # display the Dilate feed
        cv2.imshow("imgDil", imgDil)
	# display the edited image with the tracking box and x
        cv2.imshow("imagContour", imgContour)

	#To exit the program Hit esc followed by 1 then followed by q
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()

cv2.destroyAllWindows()

