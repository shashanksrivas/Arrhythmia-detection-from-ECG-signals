import pandas as pd
import biosppy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import wfdb
from keras.models import load_model,Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
import keras
from keras.utils import np_utils
from sklearn.metrics import classification_report

data=pd.read_csv("proj_mod.csv")
data = np.array(data)

labels=data[0:5100,19]
labels_test=data[5100:6000,19]
data = data[0:6000,0:19]

#signals = []
#count = 1
#peaks =  biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate = 200)[0]
#for i in (peaks[1:-1]):
#    diff1 = abs(peaks[count - 1] - i)
#    diff2 = abs(peaks[count + 1]- i)
#    x = peaks[count - 1] + diff1//2
#    y = peaks[count + 1] - diff2//2
#    signal = data[x:y]
#    signals.append(signal)
#    count += 1

#images=np.zeros([len(signals),128,128,3])
images=np.zeros([len(data),128,128,3])
#array=signals
array=data
for count, i in enumerate(array):
  fig = plt.figure(frameon=False)
  plt.plot(i) 
  plt.xticks([]), plt.yticks([])
  for spine in plt.gca().spines.values():
     spine.set_visible(False)

  filename = "" + '' + str(count)+'.png'
  fig.savefig(filename)
  im_gray = cv2.imread(filename, cv2.IMREAD_COLOR)
  im_gray = cv2.resize(im_gray, (128, 128), interpolation = cv2.INTER_LANCZOS4)
  #cv2.imwrite(filename, im_gray)
  images[count,:,:,:]=im_gray

#model = load_model('ecgScratchEpoch2.hdf5') 
IMAGE_SIZE=[128,128]

data=images[0:5100,:,:,:]
data_test=images[5100:6000,:,:,:]

labels=np_utils.to_categorical(labels,num_classes=5)
labels_test0=labels_test
labels_test=np_utils.to_categorical(labels_test,num_classes=5)

model = Sequential()

model.add(Conv2D(64, (3,3),strides = (1,1), input_shape = IMAGE_SIZE + [3],kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(64, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))

model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))

model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))

model.add(Flatten())

model.add(Dense(2048))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(data,labels,epochs=50,batch_size=10,validation_split=0.3)
  
#preds=model.predict(images)
preds=model.predict(data_test)
p=model.evaluate(data_test,labels_test)
#print(float(p[1])*100, "%")
#print(preds)
predictions=np.zeros((len(preds),1))
for count,val in enumerate(preds) :
    predictions[count]=np.argmax(val)   

print(classification_report(labels_test0,predictions))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
   