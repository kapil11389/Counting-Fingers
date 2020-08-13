# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:16:02 2020

@author: Kapil
"""


#importing required libraries

from model import *
import numpy as np,cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix


#Generating Data

train_gen=ImageDataGenerator(rotation_range=0.3,
                             horizontal_flip=True,
                             rescale=1./255,
                             shear_range=0.2,
                             zoom_range=0.5)


training_data=train_gen.flow_from_directory('./images/train',
                                            (300,300),
                                            color_mode='grayscale',
                                            batch_size=32,
                                            class_mode='categorical')


test_gen=ImageDataGenerator(rescale=1./255)

test_data=test_gen.flow_from_directory('./images/test',
                                       (300,300),
                                       color_mode='grayscale',
                                       batch_size=32,
                                       class_mode='categorical')


training_data.class_indices


#building CNN and training the model

classifier=build_model()
history = classifier.fit_generator(training_data,
                                   steps_per_epoch=100,
                                   epochs=40,
                                   validation_data=test_data,
                                   validation_steps=50)
classifier.save_weights('classifier.h5')


#Visualizing the training  and Test result

epochs=40
train_acc=history.history['accuracy']
test_acc=history.history['val_accuracy']
plt.plot(train_acc,color='blue',label='Train')
plt.plot(test_acc,color='green',label='Test')
plt.title('Model Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Epoch')
plt.legend()
plt.show()


train_loss=history.history['loss']
test_loss=history.history['val_loss']
plt.plot(train_loss,color='blue',label='Train')
plt.plot(test_loss,color='green',label='Test')
plt.title('Model Loss')
plt.xlabel('Loss')
plt.ylabel('Epoch')
plt.legend()
plt.show()



 

