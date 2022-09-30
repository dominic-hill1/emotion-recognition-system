# Import library
import cv2

# Load cascade classifier into memory
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Initialise video capture 
cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read() # take frame from webcam
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to greyscale
    faces = face_cascade.detectMultiScale( # Pass into classifier
        grayscale,
        scaleFactor=1.2,
        minNeighbours=5,
        minSize=(50,50)
    )


    for (x, y, w, h) in faces: # Iterate over faces detected 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2) # draw rectangle around face
        # roi_gray = grayscale[y:y + h, x:x + w]
        # roi_color = img[y:y + h, x:x + w]

    cv2.imshow('Face recognition', frame) # update webcam output

    k = cv2.waitKey(30) & 0xff
    if k == 27: # close if esc is pressed
        break

cap.release()
cv2.destroyAllWindows()