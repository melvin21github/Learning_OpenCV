import cv2
import numpy as np

# Load image from resource folder
image = cv2.imread('Resources/itachi.png')  

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define red color range in HSV
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# Create masks for red color (since red wraps around HSV hue)
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Bitwise-AND mask and original image
result = cv2.bitwise_and(image, image, mask=mask)

# Show results
cv2.imshow('Original', image)
cv2.imshow('Red Mask', mask)
cv2.imshow('Detected Red', result)
cv2.waitKey(0)
