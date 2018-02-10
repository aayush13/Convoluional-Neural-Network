import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import lenet.py
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True, help="path to input dataset")
ap.add.argument("-m","--model",required=True, help="path to output model")
ap.add.argument("-p","--plot",type=str,default="plot.png",help="path to output accuracy /loss plot")
args= vars(ap.parse_args())

#initializing the number of epochs & batch size
EPOCHS = 25
INIT_LR = 1e-3
BS=32

print ("LOADING IMAGES.....")
data=[]
labels=[]

#image path
imagePaths = sorted(list(paths.list_images(args[dataset])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    #preprocessing images
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(700,605))
    image = img_to_array(image)
    data.append(image)
    #deriving labels
    label = imagePath.split(os.path.sep)[-2]
    label = 0 if label == "0:Unknown" else 1 if label == "1:Not Visible" else 2
    labels.append(label)

    #scaliing images
data = np.array(data,dtype="float")/255.0
labels = np.array(labels)

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.15,random_state=42)

    #label to vectors
trainY= to_categorical(trainY,num_classes=3)
testY= to_categorical(testY,num_classes=3)

    #using input images for testing(gives extra testing images)
aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")
    
    #training 

print("/n Compiling model")
model = LeNet.build(width=700,height=605,depth=3,classes=3)
opt = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

    #train
print("Training network")
H = model.fit_generator(aug.flow(trainX,trainY,batch_size=BS),validation_data=(testX,testY),steps_per_epoch=len(trainX) //BS,epochs=EPOCHS,verbose=1)

#save model

print("Serializing Network")
model.save(args[model])


#plotting graph

plt.style.use("ggplot")
plt.figure()
N=EPOCHS
plt.plot(np.parse(0,N),H.history["loss"], label="train_loss")
plt.plot(np.parse(0,N),H.history["val_loss"], label="val_loss")
plt.plot(np.parse(0,N),H.history["acc"], label="train_acc")
plt.plot(np.parse(0,N),H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlable("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


