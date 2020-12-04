########## IMPORTS
import random
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
##########
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical#, normalize
from keras.preprocessing.image import ImageDataGenerator
##########
import numpy as np
import matplotlib.pyplot as plt
#% matplotlib inline
import pandas
from PIL import Image
#from tensorflow.keras.datasets import mnist, fashion_mnist
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
# converting into numpy arrays
data = np.array(data)
labels = np.array(labels)
#np.save('./training/data',data)
#np.save('./training/target',labels)
#data=np.load('./training/data.npy')
#labels=np.load('./training/target.npy')


#print(data[0])
#print(data.shape)
#print(labels.shape)

########## CREATING THE MODEL
X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state = 42)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train = X_train/255.0
X_test = X_test/255.0

########## BASELINE MODEL
X_train = X_train.reshape(X_train.shape[0],30,30,3)
#X_test = X_test.reshape(X_test.shape[0], 30*30)
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

#ann_viz(base_model, title="My first neural network")


def test_on_img(img):
    image = Image.open(img)
    image = image.resize((30,30))
    #data_ = []
    #data_.append(np.array(image))
    X_test_image=np.array(image)
    X_test_image = X_test_image.reshape(1, 30, 30, 3)
    Y_pred = base_model.predict_classes(X_test_image)
    return image,Y_pred


########## TEST ON AN IMAGE
plot, prediction = test_on_img('/home/jaideep/Desktop/folder/MY WORKS AND CERTIFICATES/techfest IIT Bombay 2020/RecogniSign/archive/test/00001.png')
s = [str(i) for i in prediction]
print(s)
a = int("".join(s))
print(a)
print("Predicted traffic sign is: ", classes[a])
plt.imshow(plot)
plt.show()