from __future__ import division, print_function
import json
import keras
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import pandas as pd
import numpy as np
import biosppy
import matplotlib.pyplot as plt
import wfdb
from PIL import Image
from scipy import misc
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten ,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D	 
from keras.callbacks import Callback,TensorBoard,ModelCheckpoint
from keras.regularizers import l2	
from keras.models import model_from_json
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn import datasets, linear_model
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import svm, datasets
from keras.utils import plot_model
#dataset

label = os.listdir("/home/dell/ecg/dataset_image")
dataset = []

for image_label in label:
   images = os.listdir("/home/dell/ecg/dataset_image/"+image_label)
   
   for image in images:
      img = misc.imread("/home/dell/ecg/dataset_image/"+image_label+"/"+image)
      dataset.append((img,image_label))
X = []
Y = []

for input,image_label in dataset:
   X.append(input)
   Y.append(label.index(image_label))

X = np.array(X)
Y = np.array(Y)

X_train,Y_train, = X,Y


X_train = X_train.reshape(17663,128,128,1)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.5, shuffle= True)
Y_test0=Y_test
Y_train0=Y_train
Y_train = np_utils.to_categorical(Y_train,8)
Y_test = np_utils.to_categorical(Y_test,8)



model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu',strides = (1,1), input_shape=(128,128,1)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu',strides = (1,1), input_shape=(128,128,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=3, activation='relu',strides = (1,1), input_shape=(128,128,1)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=3, activation='relu',strides = (1,1), input_shape=(128,128,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=3, activation='relu',strides = (1,1), input_shape=(128,128,1)))
model.add(BatchNormalization())


model.add(Conv2D(256, kernel_size=3, activation='relu',strides = (1,1), input_shape=(128,128,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=2, activation='relu',strides = (1,1), input_shape=(128,128,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(2048))
model.add(keras.layers.ELU())
model.add(BatchNormalization())


model.add(Dropout(rate=0.2))

model.add(Dense(8, activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=["accuracy"])

cb = TensorBoard()
history=model.fit(X_train, Y_train,batch_size=10, validation_data=(X_test, Y_test),epochs=1,verbose=1,callbacks=[cb])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

#plot_model(model, to_file='model.png')

"""
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
#model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
"""
score = model.evaluate(X_test, Y_test,batch_size=5, verbose=1)
print("Error: %.2f%%" % (100-score[1]*100))
print('loss:', score[0])
print('accuracy:', score[1])


# list all data in history
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


fig1= plt.figure()
#summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


predicted_classes = model.predict_classes(X_test)


predictions=np.zeros((len(predicted_classes),1))
for count,val in enumerate(predicted_classes):
    predictions[count]=np.argmax(val)  
print(classification_report(Y_test0,predictions))

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == Y_test0)[0]
incorrect_indices = np.nonzero(predicted_classes != Y_test0)[0]

count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
for i, correct in enumerate(correct_indices):
   if predicted_classes[correct]==0:
      count0 += 1
   elif predicted_classes[correct]==1:
      count1 += 1
   elif predicted_classes[correct]==2:
      count2 += 1
   elif predicted_classes[correct]==3:
      count3 += 1
   elif predicted_classes[correct]==4:
      count4 += 1
   elif predicted_classes[correct]==5:
      count5 += 1
   elif predicted_classes[correct]==6:
      count6 += 1
   elif predicted_classes[correct]==7:
      count7 += 1

print(count0,count1,count2,count3,count4,count5,count6,count7)
count01=0
count02=0
count03=0
count04=0
count05=0
count06=0
count07=0

count10=0
count12=0
count13=0
count14=0
count15=0
count16=0
count17=0

count20=0
count21=0
count23=0
count24=0
count25=0
count26=0
count27=0

count30=0
count31=0
count32=0
count34=0
count35=0
count36=0
count37=0

count40=0
count41=0
count42=0
count43=0
count45=0
count46=0
count47=0

count50=0
count51=0
count52=0
count53=0
count54=0
count56=0
count57=0

count60=0
count61=0
count62=0
count63=0
count64=0
count65=0
count67=0


count70=0
count71=0
count72=0
count73=0
count74=0
count75=0
count76=0

for i, incorrect in enumerate(incorrect_indices):
   if Y_test0[incorrect]==0:
      if predicted_classes[incorrect]==1:
         count01 += 1
      elif predicted_classes[incorrect]==2:
         count02 += 1
      elif predicted_classes[incorrect]==3:
         count03 += 1
      elif predicted_classes[incorrect]==4:
         count04 += 1
      elif predicted_classes[incorrect]==5:
         count05 += 1
      elif predicted_classes[incorrect]==6:
         count06 += 1
      elif predicted_classes[incorrect]==7:
         count07 += 1        
   if Y_test0[incorrect]==1:
      if predicted_classes[incorrect]==0:
         count10 += 1
      elif predicted_classes[incorrect]==2:
         count12 += 1
      elif predicted_classes[incorrect]==3:
         count13 += 1
      elif predicted_classes[incorrect]==4:
         count14 += 1
      elif predicted_classes[incorrect]==5:
         count15 += 1
      elif predicted_classes[incorrect]==6:
         count16 += 1
      elif predicted_classes[incorrect]==7:
         count17 += 1
   if Y_test0[incorrect]==2:
      if predicted_classes[incorrect]==0:
         count20 += 1
      elif predicted_classes[incorrect]==1:
         count21 += 1
      elif predicted_classes[incorrect]==3:
         count23 += 1
      elif predicted_classes[incorrect]==4:
         count24 += 1
      elif predicted_classes[incorrect]==5:
         count25 += 1
      elif predicted_classes[incorrect]==6:
         count26 += 1
      elif predicted_classes[incorrect]==7:
         count27 += 1
   if Y_test0[incorrect]==3:
      if predicted_classes[incorrect]==0:
         count30 += 1
      elif predicted_classes[incorrect]==1:
         count31 += 1
      elif predicted_classes[incorrect]==2:
         count32 += 1
      elif predicted_classes[incorrect]==4:
         count34 += 1
      elif predicted_classes[incorrect]==5:
         count35 += 1
      elif predicted_classes[incorrect]==6:
         count36 += 1
      elif predicted_classes[incorrect]==7:
         count37 += 1
   if Y_test0[incorrect]==4:
      if predicted_classes[incorrect]==0:
         count40 += 1
      elif predicted_classes[incorrect]==1:
         count41 += 1
      elif predicted_classes[incorrect]==2:
         count42 += 1
      elif predicted_classes[incorrect]==3:
         count43 += 1
      elif predicted_classes[incorrect]==5:
         count45 += 1
      elif predicted_classes[incorrect]==6:
         count46 += 1
      elif predicted_classes[incorrect]==7:
         count47 += 1
   if Y_test0[incorrect]==5:
      if predicted_classes[incorrect]==0:
         count50 += 1
      elif predicted_classes[incorrect]==1:
         count51 += 1
      elif predicted_classes[incorrect]==2:
         count52 += 1
      elif predicted_classes[incorrect]==3:
         count53 += 1
      elif predicted_classes[incorrect]==4:
         count54 += 1
      elif predicted_classes[incorrect]==6:
         count56 += 1
      elif predicted_classes[incorrect]==7:
         count57 += 1
   if Y_test0[incorrect]==6:
      if predicted_classes[incorrect]==0:
         count60 += 1
      elif predicted_classes[incorrect]==1:
         count61 += 1
      elif predicted_classes[incorrect]==2:
         count62 += 1
      elif predicted_classes[incorrect]==3:
         count63 += 1
      elif predicted_classes[incorrect]==4:
         count64 += 1
      elif predicted_classes[incorrect]==5:
         count65 += 1
      elif predicted_classes[incorrect]==7:
         count67 += 1
   if Y_test0[incorrect]==7:
      if predicted_classes[incorrect]==0:
         count70 += 1
      elif predicted_classes[incorrect]==1:
         count71 += 1
      elif predicted_classes[incorrect]==2:
         count72 += 1
      elif predicted_classes[incorrect]==3:
         count73 += 1
      elif predicted_classes[incorrect]==4:
         count74 += 1
      elif predicted_classes[incorrect]==5:
         count75 += 1
      elif predicted_classes[incorrect]==6:
         count76 += 1
         
print("no of predicted correct",len(correct_indices))

print("no of incorrect predicted",len(incorrect_indices))

print("confusion matrix")
print("images of class 0 belong to other classes",count01,count02,count03,count04,count05,count06,count07)
print("images of class 1 belong to other classes",count10,count12,count13,count14,count15,count16,count17)
print("images of class 2 belong to other classes",count20,count21,count23,count24,count25,count26,count27)
print("images of class 3 belong to other classes",count30,count31,count32,count34,count35,count36,count37)
print("images of class 4 belong to other classes",count40,count41,count42,count43,count45,count46,count47)
print("images of class 5 belong to other classes",count50,count51,count52,count53,count54,count56,count57)
print("images of class 6 belong to other classes",count60,count61,count62,count63,count64,count65,count67)
print("images of class 7 belong to other classes",count70,count71,count72,count73,count74,count75,count76)

model.summary()
