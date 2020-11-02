

from __future__ import print_function, division

from keras.layers import Dense, Flatten
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dropout
from keras.optimizers import Adam

IMAGE_SIZE = 32
num_classes = 10
batch_size = 64
epochs = 50

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=.3)

# Preprocess the data
y_train=to_categorical(y_train)
y_val=to_categorical(y_val)
y_test=to_categorical(y_test)


# Create the generator in order to easily perform data augmentation
# We rotate, flip the image and zoom into it (both training, validation and test)
train_generator = ImageDataGenerator(
                                    rotation_range=2, 
                                    horizontal_flip=True,
                                    zoom_range=.1 )

val_generator = ImageDataGenerator(
                                    rotation_range=2, 
                                    horizontal_flip=True,
                                    zoom_range=.1)

test_generator = ImageDataGenerator(
                                    rotation_range=2, 
                                    horizontal_flip= True,
                                    zoom_range=.1) 

train_generator.fit(x_train)
val_generator.fit(x_val)
test_generator.fit(x_test)

#Load the ResNet50 architecture with the pre-trained weights of imagenet
base_model_2 = ResNet50(include_top=False,weights='imagenet',input_shape=(32,32,3),classes=y_train.shape[1])

#Define the model
model_2= Sequential()
#Add the preloaded weights
model_2.add(base_model_2)
model_2.add(Flatten())

#Add new layers to the head of the network
model_2.add(Dense(4000,activation=('relu'),input_dim=512))
model_2.add(Dense(2000,activation=('relu'))) 
model_2.add(Dropout(.4))
model_2.add(Dense(1000,activation=('relu'))) 
model_2.add(Dropout(.3))
model_2.add(Dense(500,activation=('relu')))
model_2.add(Dropout(.2))

# The final layer for classification:
model_2.add(Dense(10,activation=('softmax')))


#Adam topimizer with the learning rate of 0.001
adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Training with adam optimizer and categorical crossentropy
model_2.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# Fit our model to the data we have 
model_2.fit_generator(train_generator.flow(x_train,y_train,batch_size=batch_size),
                      epochs=epochs,
                      steps_per_epoch=x_train.shape[0]//batch_size,
                      validation_data=val_generator.flow(x_val,y_val,batch_size=batch_size),validation_steps=250,
                      verbose=1)

#Plot loss and accuracy
loss1         = model_2.history.history['loss']
plt.plot(loss1)
plt.legend(['Loss'])
plt.title('Loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()


acc1         = model_2.history.history['accuracy']
plt.plot(acc1)
plt.legend(['Accuracy'])
plt.title('Accuracy ')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()
