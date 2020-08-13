# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:43:01 2020

@author: Kapil
"""

#importing required libraries

from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Dropout


#Model used in the project

def build_model():
    classifier=Sequential();
    
    classifier.add(Convolution2D(16,kernel_size=(3,3),
                                 strides=1,padding='same',
                                 activation='relu',
                                 input_shape=(300,300,1)))
    
    classifier.add(MaxPooling2D((2,2),strides=2,padding='same'))
    
    classifier.add(Convolution2D(32,kernel_size=(3,3),
                                 strides=1,padding='same',
                                 activation='relu'))
    
    classifier.add(MaxPooling2D((2,2),strides=2,padding='same'))
    
    classifier.add(Convolution2D(64,kernel_size=(3,3),
                                 strides=1,padding='same',
                                 activation='relu'))
    
    classifier.add(MaxPooling2D((2,2),strides=2,padding='same'))
    
    classifier.add(Flatten())
    
    classifier.add(Dense(units=256,activation='relu'))
    
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(Dense(units=64,activation='relu'))
    
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(Dense(units=6,activation='softmax'))
    
    classifier.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return classifier

