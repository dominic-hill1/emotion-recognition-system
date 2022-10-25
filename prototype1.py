# Import libraries
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

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

def process_image(greyscale_image):
    """
    Formats an image, ready to be passed into neural network
    """
    # Converts 1 greyscale channel into 3 identical RGB channels
    greyscale_image = cv2.cvtColor(greyscale_image,cv2.COLOR_GRAY2RGB)
    # Changes values from 0-255 to 0-1
    normalized_image = tf.image.convert_image_dtype(greyscale_image, tf.float32)
    # Changes size of image to 244x244 pixels
    formatted_image = tf.image.resize(normalized_image, size=[224, 224])

    return formatted_image

model = load_model("20220821-12521661086371-full-image-set-mobilenetv2-Adam.h5")

true_values = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load cascade classifier into memory
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Initialise video capture 
cap = cv2.VideoCapture(0)

delay_counter = 1
grey_face = False

while 1:
    ret, frame = cap.read() # take frame from webcam
    greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to greyscale
    # Pass into classifier
    faces = face_cascade.detectMultiScale(
        greyscale,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50,50)
    )


    for (x, y, w, h) in faces: # Iterate over faces detected 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2) # draw rectangle around face

        grey_face = greyscale[y:y+h, x:x+w] # crop face
    
    # print(delay_counter)


    if delay_counter % 60 == 0 and hasattr(grey_face, 'shape'):
        grey_face = process_image(grey_face)
        predictions = model.predict(np.array([grey_face]))
        print(true_values[np.argmax(predictions)])


    cv2.imshow('Face recognition', frame) # update webcam output
    delay_counter += 1

    k = cv2.waitKey(30) & 0xff
    if k == 27: # close if esc is pressed
        break

    

cap.release()
cv2.destroyAllWindows()

print(predictions)
