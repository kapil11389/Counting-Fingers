# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:33:29 2020

@author: Kapil
"""

#importing required libraries

from model import *
import cv2,numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Dropout


#removing background and detecting skin

def skin_detection(img):
    temp=cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lowerb=np.array([0,35,45])
    upperb=np.array([35,255,255])
    mask=cv2.inRange(temp, lowerb, upperb)
    img=cv2.bitwise_and(img, img,mask=mask)
    cv2.imshow('skin',img)
    cv2.waitKey(1)
    return img


#detecting the edge of hand
    
def edge_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, img = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('edge',img)
    cv2.waitKey(1)
    
    return img


#predicting the number
    
def predict(img):
   pred= classifier.predict(img)
   max_val=pred[0][0]
   index=0
   for i in range(1,6):
       if(pred[0][i]>max_val):
           index=i
           max_val=pred[0][i]
           break
   number=['Five','Four','Zero','One','Three','Two']
   return number[index]


#Loading Weights of the model
   

classifier=build_model()
classifier.load_weights('classifier.h5')


# Model visualization using opencv

x0,y0,width=150,150,300
cam=cv2.VideoCapture(0)
while(True):
    _,img=cam.read();
    img1=img[y0:y0+width,x0:x0+width]
    img1=skin_detection(img1)
    img1=edge_detection(img1)
    
    #Converting the image to the dimensionalality expected by the CNN
    img1=np.expand_dims(img1, axis=0)
    img1=np.expand_dims(img1,axis=3)
    
    pred = predict(img1)
    font=cv2.FONT_HERSHEY_COMPLEX
    img=cv2.rectangle(img, (x0,y0), (x0+width,y0+width), (0,255,0))
    img=cv2.putText(img, pred, (150,150),font,1,(0,0,255),2)
    cv2.imshow('video',img)
    if cv2.waitKey(1)==27:
        break
cam.release()
cv2.destroyAllWindows()
