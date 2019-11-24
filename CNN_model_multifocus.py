#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:14:33 2019

@author: dida
"""
import numpy as np 
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
y_val = np.load('y_val.npy')


from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D
#from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
#from keras.models import model_from_json
#from keras.models import Model
#from sklearn.model_selection import cross_val_score, cross_val_predict
#from sklearn import metrics
#from keras.models import load_model
#model = VGG16('vgg16_weights.h5')
model = VGG16(include_top=False, weights='imagenet')
#from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

weights2 = model.layers[1].get_weights()
#weights7 = model.layers[7].get_weights()
#weights12 = model.layers[12].get_weights()


classifier = Sequential()
conv1 = Conv2D(64, (3, 3),strides=(1, 1), padding='valid', input_shape = (7, 7, 3))
classifier.add(conv1)
classifier.add(Activation('relu'))
#classifier.set_weights(weights2)
conv2 = Conv2D(128, (3, 3),strides=(1, 1), padding='valid', input_shape = (5, 5, 3))
classifier.add(conv2)
classifier.add(Activation('relu'))
#classifier.set_weights(weights7)
classifier.add(Flatten())
classifier.add(Dense(units = 1152))
classifier.add(Dense(units = 2))
classifier.add(Activation('softmax'))    

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

history = classifier.fit(X_train, y_train,
              batch_size=32,
              epochs=20,
              validation_data=(X_val, y_val),
              shuffle=True)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['val_categorical_accuracy'])
plt.plot(history.history['categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
cvscores = []
#print("TRAIN:",X_train, "VALIDATION:", X_validate)
scores = classifier.evaluate(X_val, y_val)
print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
cvscores.append(scores[1] * 100)
        
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))    




# SAVING
model_json = classifier.to_json()
with open("LAST.json", "w") as json_file:
    json_file.write(model_json)
    
from keras.utils import plot_model
import pydot
pydot.find_graphviz()
plot_model(classifier,show_shapes=True, to_file='LAST.png')    
# serialize weights to HDF5
classifier.save_weights("LAST.h5");