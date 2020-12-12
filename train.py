# CODE THAT IMPORTS DATA, PREPROCESSES IT, CREATES THE NN MODEL AND TRAINS IT ON DATA.

########## IMPORTING NECESSARY LIBRARIES
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import pandas
from PIL import Image
from sklearn.model_selection import train_test_split
from ann_visualizer.visualize import ann_viz

########## DATASET IMPORT AND PREPROCESSING
classes = 43
data = []
labels = []
cur_path = os.getcwd()
path = os.path.join(cur_path,'archive/train')

for i in range(classes):
    new_path = os.path.join(path,str(i))
    images = os.listdir(new_path)
    for a in images:
        try:
            image = Image.open(new_path +'/'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)
# Converting into numpy arrays
data = np.array(data)
labels = np.array(labels)


#print(data[0])
#print(data.shape)
#print(labels.shape)

########## CREATING DATA SPLIT, USING MAXIMUM DATA FOR TRAINING
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.01, random_state = 42)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train = X_train/255.0
X_test = X_test/255.0

########## BASELINE MODEL
X_train = X_train.reshape(X_train.shape[0],30,30,3)
base_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(1,30,30,3),),
    tf.keras.layers.Dense(128, activation='relu', input_shape=(1,30,30,1)),
    tf.keras.layers.Dense(43)
])

base_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

base_model.fit(X_train, y_train, epochs=20)

########## SAVE MODEL
base_model.save("traffic_model")

test_loss, test_acc = base_model.evaluate(X_test,  y_test, verbose=2)
print('Test accuracy:', test_acc)

########## CLASSES OF TRAFFIC SIGNS
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


########## TEST ON AN IMAGE
def test_on_img(img):
    image = Image.open(img)
    image = image.resize((30,30))
    X_test_image=np.array(image)
    X_test_image = X_test_image.reshape(1, 30, 30, 3)
    Y_pred = base_model.predict_classes(X_test_image)
    return image,Y_pred