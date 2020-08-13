# CNN Counting-Fingers
A CNN deep learning model that counts the number of fingers held up by the hand using Keras , TensorFlow as backend and OpenCV

# Requirments

Dataset used can be downloaded from : https://github.com/jaredvasquez/CNN-HowManyFingers

NVIDIA 940MX GPU was used for speedy computation when training the model 

Python 3.5+ , Keras 2+ , Tensorflow 2+, OpenCV 4.3+

# Contents
model.py : This file contains the CNN deep learning model used in the project

CountingFingers.py : This file contains the code to generate the data and training the model

Result_Demo.py : An application designed using OpenCV to analyse the performance of the model

# Training Data and Test Data

The Images are 300\*300 size black and white Images

![Figure_1](https://user-images.githubusercontent.com/41421032/90111452-4f9b3680-dd6c-11ea-96f5-4de070e51836.png)

# Reading Data from directory and reult mapping
![im](https://user-images.githubusercontent.com/41421032/90111292-0f3bb880-dd6c-11ea-8dc4-fb6882ff1ac7.PNG)
![class](https://user-images.githubusercontent.com/41421032/90111921-f2ec4b80-dd6c-11ea-910a-eadd655fbdb9.PNG)

# Model 

![summary](https://user-images.githubusercontent.com/41421032/90111542-6b064180-dd6c-11ea-95c3-1f7866925eb0.PNG)

# Training the model

![training](https://user-images.githubusercontent.com/41421032/90111612-85d8b600-dd6c-11ea-88e7-e9efc3e638d4.PNG)

# Training and Test result Visualization 

![Figure_2](https://user-images.githubusercontent.com/41421032/90111757-ba4c7200-dd6c-11ea-9146-b8a348850b4c.png)

![Figure_1](https://user-images.githubusercontent.com/41421032/90111770-bcaecc00-dd6c-11ea-8151-0a23ee0a8fbb.png)

# Model Demo
![test](https://user-images.githubusercontent.com/41421032/90112010-131c0a80-dd6d-11ea-8918-3b3ba95dfcd7.png)

# Observations
1. The model gives accuracy > 90% and many a times beats human vision.
2. The Model surpisingly gives correct results on the combination of fingers held never seen by it.

I am sure better resluts can be obtained by adding more number of hidden layers and CNN layers and more number of neurons.
High Quality images can  also be used to enhance the capabilities of the model.
