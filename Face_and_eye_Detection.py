import cv2

# Load the cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Resources/haarcascade_eye.xml')
# Check if it loaded correctly
if face_cascade.empty():
    print(" Failed to load Haar cascade for face!")
    exit()
if eye_cascade.empty():
    print(" Failed to load Haar cascade for eye!")
    exit()
# Read the input image
img = cv2.imread('Resources/3people.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# Display the output
img = cv2.resize(img,(900,700))
cv2.imshow('Face and Eye Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()