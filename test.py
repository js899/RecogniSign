# TO TEST ON A SELECTED IMAGE. PUT THE PATH OF THE IMAGE IN LINE 20.

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
########## MODEL IMPORT
test_model = tf.keras.models.load_model("traffic_model")

########## TEST ON AN IMAGE

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

def test_on_img(img):
    image = Image.open(img)
    image = image.resize((30,30))
    X_test_image=np.array(image)
    X_test_image = X_test_image.reshape(1, 30, 30, 3)
    Y_pred_pc = test_model.predict(X_test_image)
    probval = np.max(Y_pred_pc)
    Y_pred = test_model.predict_classes(X_test_image)
    return image, Y_pred, probval

plot, prediction, probvalue = test_on_img('/home/jaideep/Desktop/folder/MY WORKS AND CERTIFICATES/techfest IIT Bombay 2020/RecogniSign/archive/test/00001.png')
s = [str(i) for i in prediction]
print(s)
a = int("".join(s))
print(a)
print("Predicted traffic sign is: ", classes[a] +" "+ str(round(probvalue, 2))+"%")
plt.imshow(plot)
plt.show()