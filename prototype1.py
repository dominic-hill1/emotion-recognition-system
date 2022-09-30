# Import libraries
import cv2
import tensorflow as tf
import tensorflow_hub as hub

# Check library versions
print("Tensorflow version= {}".format(tf.__version__))
print("Tensorflow Hub version= {}".format(hub.__version__))

# Load model
def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer": hub.KerasLayer})
  return model

model = load_model("20220821-12521661086371-full-image-set-mobilenetv2-Adam.h5")

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
        minNeighbors=5,
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