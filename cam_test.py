import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from test import imgpreprocess
from test import classname
from main import base_model

########## RESOLUTION PARAMETERS
frameWidth = 640
frameHeight = 640
brightness = 100
threshold = 0.80
font = cv2.FONT_HERSHEY_SIMPLEX

########## CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(5, brightness)

########## MODEL LOAD
#test_model = tf.keras.models.load_model("traffic_model")

while True:
    success, image_seen = cap.read()
    image_ = np.asarray(image_seen)
    image_ = cv2.resize(image_, (30, 30))
    image_ = imgpreprocess(image_)
    cv2.imshow("Processed Image", image_)
    image_ = image_.reshape(1, 30, 30, 3)
    cv2.putText(image_seen, "CLASS: ", (20,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(image_seen, "PROBABILITY: ", (20,75), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    predictions = base_model.predict(image_)
    probval = np.max(predictions)
    class_index = np.argmax(predictions)
    if probval > threshold:
       cv2.putText(image_seen, "CLASS: "+str(class_index)+str(classname(class_index)), (120,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
       cv2.putText(image_seen, "PROBABILITY: "+str(round(probval, 2))+"%", (180,75), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Result", image_seen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()