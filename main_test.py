import os
import tensorflow as tf
import cv2
import numpy as np

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

########## MODEL IMPORT
test_model = tf.keras.models.load_model("traffic_model")

def imgpreprocess(img):
    # to grayscale
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # equalize
    #img = cv2.equalizeHist(img)
    img = img/225.0
    return img

def classname(class_index):
    classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
    value = classes.get(class_index)
    return value

while True:
    success, image_seen = cap.read()
    image_ = np.asarray(image_seen)
    image_ = cv2.resize(image_, (30, 30))
    image_ = imgpreprocess(image_)
    cv2.imshow("Processed Image", image_)
    image_ = image_.reshape(1, 30, 30, 3)
    cv2.putText(image_seen, "CLASS: ", (20,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(image_seen, "PROBABILITY: ", (20,75), font, 0.75, (0,0,255), 2, cv2.LINE_AA)

     # PREDICT IMAGE USING WEBCAM
    predictions = test_model.predict(image_)
    #class_index = np.argmax(test_model.predict_classes(image_), axis=-1)
    probval = np.max(predictions)
    class_index = np.argmax(predictions)
    #print(probval)
    if probval > threshold:
       cv2.putText(image_seen, "CLASS: "+str(class_index)+str(classname(class_index)), (120,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
       cv2.putText(image_seen, "PROBABILITY: "+str(round(probval, 2))+"%", (180,75), font, 0.75, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Result", image_seen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()