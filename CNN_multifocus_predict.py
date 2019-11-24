#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:21:37 2019

@author: dida
"""

import scipy
import scipy.io as sio
from keras.models import model_from_json
json_file = open("VGG16dummy_split.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("VGG16dummy_split.h5")
print("Loaded model from disk")

#OPTION 2
#classifier.save('myModel.hdf5')
#loaded_model=load_model('myModel.hdf5')

# evaluate loaded model on test data
#loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(None, None, verbose=0,steps=None)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

import numpy as np
import cv2
import os
import time
import glob
# Load an color image in grayscale
img1= cv2.imread('C:/Users/dida/Desktop/SXOLI/mf_dataset_clean/mis/mis_8/lab1.bmp',0)
img2= cv2.imread('C:/Users/dida/Desktop/SXOLI/mf_dataset_clean/mis/mis_8/lab2.bmp',0)



#image_list1 = []
#image_list2 = []
#for i in range(42,52,1):
# for filename1 in glob.glob('C:/Users/dida/Desktop/SXOLI/mf_Pascal2/pascal_'+str(i)+'/'+'x1.png'): 
#     img1_list = cv2.imread(filename1,0)
#     image_list1.append(img1_list)
# for filename2 in glob.glob('C:/Users/dida/Desktop/SXOLI/mf_Pascal2/pascal_'+str(i)+'/'+'x2.png'): 
#     img2_list = cv2.imread(filename2,0)
#     image_list2.append(img2_list)



    #cv2.imshow('image',img1_list)
    #cv2.waitKey(500)
    #cv2.destroyAllWindows()
    #cv2.imshow('image',img2_list)
    #cv2.waitKey(500)
    #cv2.destroyAllWindows()
    
#for k in range(0 , np.size(image_list1,0),1)  :
start = time.time()
#img1 = image_list1[k]
#img2 = image_list2[k]



    #img1_orgnl = np.array(img1_orgnl, dtype=np.uint8)
    #img2_orgnl = np.array(img2_orgnl, dtype=np.uint8)
    #img1 = cv2.cvtColor(img1_orgnl, cv2.COLOR_BGR2GRAY)
    #img2 = cv2.cvtColor(img2_orgnl, cv2.COLOR_BGR2GRAY)


height = np.size(img1,0)
width = np.size(img1,1)
I = np.zeros((height,width,3), np.uint8)
I[:,:,0]=img1
I[:,:,1]=img2
I[:,:,2]=0
I = I/255
classes = np.zeros((height,width), np.uint8)
binary = np.zeros((height,width,3), np.uint8)
#color = np.zeros((height,width,3), np.uint8)
fused = np.zeros((height,width,3), np.uint8)
probs = []
cv2.imshow('image',img1)
cv2.waitKey(5000)
cv2.destroyAllWindows() 
sheight=7
swidth=7
counter = 0
for i in range(0,height-sheight,1):
    for j in range(0,width-swidth,1):
      Icrp=I[i:i+sheight,j:j+swidth]      
#path1 = 'C:\\Users\\dida\\Desktop\\example\\inputs\\'
#cv2.imwrite(os.path.join(path1 , "%d.png"%counter), Icrp)
#img = cv2.imread('inputs/*.png',1)
#classifier.predict(np.expand_dims(img, axis=0))    
#loaded_model.predict_classes(165,Icrp)
      #img = cv2.imread('22.png')
      prediction_proba = loaded_model.predict_proba(Icrp.reshape(1,sheight,swidth,3))
      prediction_classes = loaded_model.predict_classes(Icrp.reshape(1,sheight,swidth,3))
      probs.extend(prediction_proba)
      #print(prediction_proba)  

      #binary[i,j] =  prediction_classes
      #binary[binary==1] = 255
      #fused[i,j] = I[i,j,prediction_classes]
      #if prediction_classes == 0 :
       #     fused[i,j] = img1[i,j]
      ##else:
           #fused[i,j] = img2[i,j]

      classes[i:i+sheight,j:j+swidth] = classes[i:i+sheight,j:j+swidth]+prediction_classes
      if np.nanmean(classes[i:i+sheight,j:j+swidth])<23:
           binary[i:i+sheight,j:j+swidth,:] = 0
           #binary[i:i+sheight,j:j+swidth,:].reshape(1,sheight,swidth,3)
           fused[i:i+sheight,j:j+swidth,0] = img1[i:i+sheight,j:j+swidth] 
           fused[i:i+sheight,j:j+swidth,1] = img1[i:i+sheight,j:j+swidth]
           fused[i:i+sheight,j:j+swidth,2] = img1[i:i+sheight,j:j+swidth]
      else:
           binary[i:i+sheight,j:j+swidth,:] = 255
           #binary[i:i+sheight,j:j+swidth,:].reshape(1,sheight,swidth,3)
           fused[i:i+sheight,j:j+swidth,0] = img2[i:i+sheight,j:j+swidth] 
           fused[i:i+sheight,j:j+swidth,1] = img2[i:i+sheight,j:j+swidth]
           fused[i:i+sheight,j:j+swidth,2] = img2[i:i+sheight,j:j+swidth]
        #""""probs1 = []
        #for k in range(i*j):
        #    probs1.extend(probs[k])
        #myarray = np.asarray(probs1)
        #print(probs1)
        #probsarray = myarray.reshape(2*i,j)
        #print(probsarray)"""
        #rgb = cv2.cvtColor(fused,cv2.COLOR_GRAY2BGR)    
        #cv2.imshow('image',binary)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows() 
        #cv2.imshow('image',fused)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
path1 = 'C:/Users/dida/Desktop/example/all_binary2'
path2 = 'C:/Users/dida/Desktop/example/all_fused2'
cv2.imwrite(os.path.join(path1 ,'mis8.png'), binary)
cv2.imwrite(os.path.join(path2 ,'mis8.png'), fused)
end = time.time()
print('TIME = ', (end-start))
import matplotlib.pyplot as plt
#plotbinary = plt.imshow(binary)
#plotfused = plt.imshow(fused)
probas = np.array(probs)
probas = np.reshape(probs,(height-sheight,width-swidth,2))
from keras.utils import plot_model
scipy.io.savemat('probas2', {"probas":probas})

#from sklearn.metrics import mean_squared_error
#from skimage.measure import compare_ssim
#orgnal = cv2.imread('ground.jpg')
#gray_orgnal = cv2.cvtColor(orgnal,cv2.COLOR_BGR2GRAY)
#mse = mean_squared_error(fused, gray_orgnal)
#ssim = compare_ssim(fused, gray_orgnal)
#print(mse)
#print(ssim)

pred_img_c1 = np.zeros((height,width), np.uint8)
pred_img_c2 = np.zeros((height,width), np.uint8)
k=0
for x in range(0,height-sheight,1):
    for y in range(0,width-swidth,1):
        pred_img_c1[x,y] = probs[k][0]*255
        pred_img_c2[x,y] = probs[k][1]*255
        k = k+1
cv2.imwrite(os.path.join(path1 ,'58_probas1.png'), pred_img_c1)
cv2.imwrite(os.path.join(path2 , '58_probas2.png'), pred_img_c2)

plotpred_img_c1 = plt.imshow(pred_img_c1)
plotpred_img_c2 = plt.imshow(pred_img_c2)

