# imports
import face_recognition
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import models

# loading model
# model = keras.models.load_model("my_model")
model = keras.models.load_model("my_model_2")

# classmates = ['Elayne', 'Alejandra', 'Greg', "A'Darius", 'Danny', 'Maria', 
#           'Lizbet', 'Giancarlos', 'Julio', 'Abraham', 'Alexis', 'Claudia', 
#           'Sam', 'Jonathan', 'Ron', "N'Kira", 'Jose', 'Trinity', 'Wren', 
#           'Steve', 'Marvin']
classmates = ["A'Darius", 'Alejandra', 'Claudia', 'Sam']

# helper functions
def fixImg(img, rotate = True):
  img = img[:,:,::-1] # Color correction
  if rotate:
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE) # Rotation
  return img

def crop_to_face(img):
    face_locations = face_recognition.face_locations(img)
    output_imgs = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        new_img = img[top:bottom,left:right,:] # Cropping
        new_img = cv2.resize(new_img, (64,64)) # Resizing
        output_imgs.append(new_img)
    return output_imgs
  
def draw_boxes_on_faces(img, predictions):
    face_locations = face_recognition.face_locations(img)
    for face_idx, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        img = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        prediction = [classmates[_] for _ in predictions[face_idx].argsort()[-1:][::-1]][0]
        confidence = sorted(predictions[face_idx])[-1:][::-1][0]
        cv2.putText(img, prediction + ' ' + str(confidence), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

# webcam display
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while(True):
    
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = crop_to_face(frame)
    faces = np.array(faces, dtype=np.float64)
    faces = (faces - faces.mean()) / faces.std()
    if len(faces) == 0:
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      cv2.imshow('face', frame)
    else: 
      predictions = model.predict(np.array(faces) )
      display = draw_boxes_on_faces(frame, predictions)
      display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      cv2.imshow('face', display)
      
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()