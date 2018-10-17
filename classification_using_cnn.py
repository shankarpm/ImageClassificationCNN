# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
# to add the convolution layers
from keras.layers import Conv2D
# to add the pooling layers
from keras.layers import MaxPooling2D
#to do flattening 
#in which we convert all the pooled features maps that we created thru
#convolution and maxpooling into this large feature vector
#that is then becoming the input of the fully connected layers
from keras.layers import Flatten
#we use this package to add the fully connected layers in a classic
#ANN
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# we create many feature maps to obtain our first convolution layer
# we apply feature detector to the input image and create a feature
#map. creating the feature map from the input image using feature
# detector is the convolution operation.
# In this step , we create many feature detector.
# there will many many feature detector and apply to the input image
# and get many feature maps to obtain our first convolution layer


#we will add 32 feature detectors and 3 and 3 as dimension
#input image as (color image 3d array) - so we put 3 for color and 64 by 64 bit for image pixel.
#In this example , we are tensorflow back end.
#
#make sure we dont have negative values in the feature map.
#we need to remove negative values inorder to have non-linearity in the CNN
#classifying images is a non- linearity problem and so we need to have non-linearity in our model
#so we use (relu) rectiifier activation function to achieve non-linearity.
classifier.add(Conv2D(32, (3, 3), 
                      input_shape = (64, 64, 3),
                      activation = 'relu'))

# Step 2 - Pooling
#MaxPooling- taking the max value from feature map to Pooled feature map
#pool_size = default size 2 by 2 to scan and get the max from feature map. this is recommeded value.

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening

#flattening - flattening the above maxpool map to a flat vector.
classifier.add(Flatten())

# Step 4 - Full connection

#common practice is to choose a number of hidden nodes between no of input nodes and output nodes.
#common practice to choose output_dim as 128 in the hidden layer is based on the experiment. it is based on trial n error.
#Sigmoid function is used when u need a binary outcome.
#last layer which is the output layer , for that output_dim shud be 1 , because there is only one output.

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
#since its a binary outcome we use binary_crossentropy as loss output.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
#image augmentation is done on the training set data to prevent overfitting in the model.
#it changes the images by rotating and transforming it and does training on the model.
#image augmentation is a technique that alows us to enrich our datasets without addding more images and 
#therefore that allows to get good performance results with little or no overfitting.
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


#we can improve the accuracy of test set by adding a convolution layer or full connected NN layer in the ANN
#or we can increase it by increasing the feature detectors
#also we can increase target size of image pixel in test_datagen.flow_from_directory() function to increase accuracy
#
#the accuracy we got here is around 84%

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/tumorSample.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Found'
else:
    prediction = 'NotFound'