import cv2
import numpy as np

img = cv2.imread("Resources/itachi.png",0)

cv2.imshow("Output",img)
cv2.waitKey(0)