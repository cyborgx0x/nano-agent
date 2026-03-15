# TechVidvan Object detection of similar color

import cv2
import numpy as np

# Reading the image
img = cv2.imread('image.jpg')

# Showing the output

# cv2.imshow("Image", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# convert to hsv colorspace
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



lower_bound = np.array([98,130,140]) 
upper_bound = np.array([139,148,255])

# find the colors within the boundaries
mask = cv2.inRange(hsv, lower_bound, upper_bound)

#define kernel size  
kernel = np.ones((7,7),np.uint8)

# Remove unnecessary noise from mask

# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Segment only the detected region
# segmented_img = cv2.bitwise_and(img, img, mask=mask)

# Find contours from the mask

# contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# converting image into grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# setting threshold of gray image
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(threshold,
    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
i = 0
for contour in contours:
    if i == 0:
        i = 1
        continue
  
    # cv2.approxPloyDP() function to approximate the shape
    area = cv2.contourArea(contour)
    if area > 0 and area < 1000: 
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:

        # using drawContours() function
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
    
        # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
# output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

# output = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
# Showing the output
            output = cv2.drawContours(img, contour, -1, (0, 0, 255), 3)

cv2.imshow("Output", output)
cv2.waitKey(0)
