from __future__ import print_function

#The two folloing lines allow to reduce tensorflow verbosity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1' # '0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING msgs, '3' to filter all msgs

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models 
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import numpy as np


print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)

##Uncomment the following two lines if you get CUDNN_STATUS_INTERNAL_ERROR initialization errors.
## (it happens on RTX 2060 on room 104/moneo or room 204/lautrec) 
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)



#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
#
# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8
print('x_train.shape=', x_train.shape)
print('y_test_.shape=', y_test_.shape)

num_pixels = x_train.shape[1] * x_train.shape[2]
x_train_3  = x_train.reshape(x_train.shape[0], num_pixels)
x_test_3   = x_test.reshape(x_test.shape[0], num_pixels)

#To input our values in our network Conv2D layer, we need to reshape the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 28, 28, 1) where 1 is the number of channels of our images
# =============================================================================

# =============================================================================
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#  
# =============================================================================
# =============================================================================
#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255


# =============================================================================
# print('x_train.shape=', x_train.shape)
# print('x_test.shape=', x_test.shape)
# 
# =============================================================================

num_classes = 10

#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = tensorflow.keras.utils.to_categorical(y_train_, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test_, num_classes)
# num_classes is computed automatically here
# but it is dangerous if y_test has not all the classes
# It would be better to pass num_classes=np.max(y_train)+1



# =============================================================================
# plt.imshow(x_train[1], cmap=plt.cm.binary)
# plt.axis('off')
# plt.show()
# plt.savefig('0' + '.png')
# 
# plt.imshow(x_train[3], cmap=plt.cm.binary)
# plt.show()
# plt.savefig('1' + '.png')
# 
# plt.imshow(x_train[5], cmap=plt.cm.binary)
# plt.show()
# plt.savefig('2' + '.png')
# 
# plt.imshow(x_train[7], cmap=plt.cm.binary)
# plt.show()
# plt.savefig('3' + '.png')
# 
# plt.imshow(x_train[2], cmap=plt.cm.binary)
# plt.show()
# plt.savefig('4' + '.png')
# 
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()
# plt.savefig('5' + '.png')
# 
# plt.imshow(x_train[13], cmap=plt.cm.binary)
# plt.show()
# plt.savefig('6' + '.png')
# 
# plt.imshow(x_train[15], cmap=plt.cm.binary)
# plt.show()
# plt.savefig('7' + '.png')
# 
# plt.imshow(x_train[17], cmap=plt.cm.binary)
# plt.show()
# plt.savefig('8' + '.png')
# 
# plt.imshow(x_train[4], cmap=plt.cm.binary)
# plt.show()
# plt.savefig('9' + '.png')
# =============================================================================

# =============================================================================
# class_names = ['0', '1', '2', '3', '4',
#                 '5', '6', '7', '8', '9']
# 
# plt.figure(figsize=(5,3))
# for i in range(15):
#     plt.subplot(3,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     
#     plt.xlabel(class_names[y_train[i]])
# plt.show()
# =============================================================================


##### Model 1 #####

# Define Model Type
model1 = models.Sequential()

# Add single convolutional layer 
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))

# Flatten into one dimension feature vector per image sample
model1.add(layers.Flatten())

# Add a fully connected layer
model1.add(layers.Dense(10, activation='softmax'))

# Compile the model with best optimizer
model1.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 25 epochs
history = model1.fit(x_train, y_train, epochs=25, validation_split=0.2)



##### Model Lab3 #####

# Define model type
model_lab3 = models.Sequential()

# Add a single hidden layer with the best activation function found in lab 3
model_lab3.add(Dense(64 ,activation = 'relu'))

# Add last fully connected layer with categorical activation function
model_lab3.add(Dense(num_classes, activation='softmax'))

# Compile the model given the best optimizer
model_lab3.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model for 50 epochs given the optimal batch size of 1024
history_lab3 = model_lab3.fit(x_train_3, y_train, epochs= 25, batch_size = 1024, validation_split=0.2)


# Evaluate both models on test set
testloss_1, testaccuracy_1       = model1.evaluate(x_test, y_test)
testloss_lab3, testaccuracy_lab3 = model_lab3.evaluate(x_test_3, y_test)


# Fetching the losses for both models to be compared
loss1         = history.history['loss']
val_loss1     = history.history['val_loss']
loss_lab3     = history_lab3.history['loss']
val_loss_lab3 = history_lab3.history['val_loss']

#Plotting the losses
plt.plot(loss1)
plt.plot(val_loss1)
plt.plot(loss_lab3)
plt.plot(val_loss_lab3)

# Formatting the plot and saving it if desired
plt.legend(['model 1', 'model 1 val', 'model lab 3', 'model lab 3 val'])
plt.title('Loss and Validation Loss')
plt.xlabel('iteration')
plt.ylabel('loss')
#plt.savefig('MNIST_losses_SingleConv_LastLab3' + '.png')
plt.show()


# Fetching the accuracies for both models to be compared
acc1         = history.history['accuracy']
val_acc1     = history.history['val_accuracy']
acc_lab3     = history_lab3.history['accuracy']
val_acc_lab3 = history_lab3.history['val_accuracy']

#Plotting the accuracies
plt.plot(acc1)
plt.plot(val_acc1)
plt.plot(acc_lab3)
plt.plot(val_acc_lab3)

#Formatting the plot and saving it if desired
plt.legend(['model 1', 'model 1 val', 'model lab 3', 'model lab 3 val'])
plt.title('Accuracy and Validation Accuracy')
plt.xlabel('iteration')
plt.ylabel('accuracy')
#plt.savefig('MNIST_accuracy_SingleConv_LastLab3' + '.png')
plt.show()


##### Model 2 #####

# Define Model Type
model2 = models.Sequential()

# Add fist convolutional layer 
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))

# Max Pooling between convolutional layers
model2.add(layers.MaxPooling2D((2, 2)))

# Add second convolutional layer
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten into one dimension feature vector per image sample
model2.add(layers.Flatten())

# Add a fully connected layer
model2.add(layers.Dense(10, activation='softmax'))

# Compile the model with best optimizer
model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 25 epochs
history2= model2.fit(x_train, y_train, epochs=25, validation_split=0.2)



##### Model 3 #####

# Define Model Type
model3 = models.Sequential()

# Add fist convolutional layer 
model3.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))

# Max Pooling between convolutional layers
model3.add(layers.MaxPooling2D((2, 2)))

# Add second convolutional layer 
model3.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Max Pooling between convolutional layers
model3.add(layers.MaxPooling2D((2, 2)))

# Add third convolutional layer 
model3.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Flatten into one dimension feature vector per image sample
model3.add(layers.Flatten())

# Add a fully connected layer
model3.add(layers.Dense(10, activation='softmax'))

# Compile the model with best optimizer
model3.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 25 epochs
history3= model3.fit(x_train, y_train, epochs=25, validation_split=0.2)


# EValuate three convolutional neural networks on test set
testloss_1, testaccuracy_1 = model1.evaluate(x_test, y_test)
testloss_2, testaccuracy_2 = model2.evaluate(x_test, y_test)
testloss_3, testaccuracy_3 = model3.evaluate(x_test, y_test)

# Fetching the losses for all three models to be compared
loss1 = history.history['loss']
val_loss1 = history.history['val_loss']
loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']
loss3 = history3.history['loss']
val_loss3 = history3.history['val_loss']

#Plotting the losses
plt.plot(loss1)
plt.plot(val_loss1)
plt.plot(loss2)
plt.plot(val_loss2)
plt.plot(loss3)
plt.plot(val_loss3)

#Formatting the plot and saving it if needed
plt.legend(['model 1', 'model 1 val', 'model 2', 'model 2 val', 'model 3', 'model 3 val'])
plt.title('Loss and Validation Loss')
plt.xlabel('iteration')
plt.ylabel('loss')
#plt.savefig('MNIST_losses_3models_25epochs' + '.png')
plt.show()
#plt.close()

# Fetching the accuracies for all three models to be compared
acc1 = history.history['accuracy']
val_acc1 = history.history['val_accuracy']
acc2 = history2.history['accuracy']
val_acc2 = history2.history['val_accuracy']
acc3 = history3.history['accuracy']
val_acc3 = history3.history['val_accuracy']


#Plotting the accuracies
plt.plot(acc1)
plt.plot(val_acc1)
plt.plot(acc2)
plt.plot(val_acc2)
plt.plot(acc3)
plt.plot(val_acc3)

#Formatting the plot and saving it if needed
plt.legend(['model 1', 'model 1 val', 'model 2', 'model 2 val', 'model 3', 'model 3 val'])
plt.title('Accuracy and Validation Accuracy')
plt.xlabel('iteration')
plt.ylabel('accuracy')
#plt.savefig('MNNIST_accuracy_3models_25epochs' + '.png')
plt.show()
plt.close()
  
#Defining the class names
class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']

# =============================================================================
# # Creating paths for new directories
# path = os.getcwd()
# path_1 = path+"\\model1"
# path_lab3 = path+"\\model_lab3"
# path_2 = path+"\\model2"
# path_3 = path+"\\model3"
# 
# # Checking existence or creating directories for report images
# if os.path.exists(path_1):
#     print(path_1 + ' : exists')
# else:
#     print(path_1 + ' : created')
#     os.mkdir(path_1)
# 
# if os.path.exists(path_lab3):
#     print(path_lab3 + ' : exists')
# else:
#     print(path_lab3 + ' : created')
#     os.mkdir(path_lab3)
# 
# if os.path.exists(path_2):
#     print(path_2 + ' : exists')
# else:    
#     print(path_2 + ' : created')
#     os.mkdir(path_2)
# 
# if os.path.exists(path_3):
#     print(path_3 + ' : exists')
# else:
#     print(path_3 + ' : created')
#     os.mkdir(path_3)
# =============================================================================


##### Model 1 #####
# Predicting probability outputs for test data
y_prob = model1.predict(x_test)

# Obtaining the highest probabilities, the 'most likely' to be correct class
y_predicted = y_prob.argmax(axis=-1)

#Obtaining the correctly and incorrectly classified class vs the ground truth
y_test_=y_test_.reshape((-1,))
incorrects = np.nonzero(y_predicted != y_test_)
corrects = np.nonzero(y_predicted == y_test_)
y_prob[corrects] = 0
prob_incorrects= (np.max(y_prob,axis=1))

#Obtain the number of incorrectly classified instances
n_incorrects  = np.sum(prob_incorrects!=0)
print('Number of incorrecty classified numbers on test set, for model 1: ' + str(n_incorrects))

#Get the worst 10 classified instances
top_wrong = np.argsort(prob_incorrects)
top_wrong = np.flip(top_wrong)
if n_incorrects<10:
    top_n = top_wrong[0:n_incorrects]
else:
    top_n = top_wrong[0:10]

#Plotting wrongly classified instances and saving them if needed
for i in range(np.size(top_n)):
    plt.imshow(x_test[top_n[i]], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)    
    plt.xlabel('Correct: '+class_names[y_test_[top_n[i]]]+' / Predicted:'+class_names[y_predicted[top_n[i]]])
    #plt.savefig(path_1+'/'+str(i+1)+'_Model1_Correct'+class_names[y_test_[top_n[i]]]+'_Predicted'+class_names[y_predicted[top_n[i]]] +'.jpg' )
    plt.show()
    

##### Lab 3.3 model #####
# Predicting probability outputs for test data
y_prob = model_lab3.predict(x_test_3)

# Obtaining the highest probabilities, the 'most likely' to be correct class
y_predicted = y_prob.argmax(axis=-1)

#Obtaining the correctly and incorrectly classified class vs the ground truth
y_test_=y_test_.reshape((-1,))
incorrects = np.nonzero(y_predicted != y_test_)
corrects = np.nonzero(y_predicted == y_test_)
y_prob[corrects] = 0
prob_incorrects= (np.max(y_prob,axis=1))

#Obtain the number of incorrectly classified instances
n_incorrects  = np.sum(prob_incorrects!=0)
print('Number of incorrecty classified numbers on test set, for model lab 3.3: ' + str(n_incorrects))

#Get the worst 10 classified instances
top_wrong = np.argsort(prob_incorrects)
top_wrong = np.flip(top_wrong)
if n_incorrects<10:
    top_n = top_wrong[0:n_incorrects]
else:
    top_n = top_wrong[0:10]


#Plotting wrongly classified instances and saving them if needed
for i in range(np.size(top_n)):
    plt.imshow(x_test[top_n[i]], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)    
    plt.xlabel('Correct: '+class_names[y_test_[top_n[i]]]+' / Predicted:'+class_names[y_predicted[top_n[i]]])
    #plt.savefig(path_lab3+'/'+str(i+1)+'_Model_Lab3_Correct'+class_names[y_test_[top_n[i]]]+'_Predicted'+class_names[y_predicted[top_n[i]]] +'.jpg' )
    plt.show()

##### Model 2 #####
# Predicting probability outputs for test data
y_prob = model2.predict(x_test)

# Obtaining the highest probabilities, the 'most likely' to be correct class
y_predicted = y_prob.argmax(axis=-1)

#Obtaining the correctly and incorrectly classified class vs the ground truth
y_test_=y_test_.reshape((-1,))
incorrects = np.nonzero(y_predicted != y_test_)
corrects = np.nonzero(y_predicted == y_test_)
y_prob[corrects] = 0
prob_incorrects= (np.max(y_prob,axis=1))

#Obtain the number of incorrectly classified instances
n_incorrects  = np.sum(prob_incorrects!=0)
print('Number of incorrecty classified numbers on test set, for model 2: ' + str(n_incorrects))

#Get the worst 10 classified instances
top_wrong = np.argsort(prob_incorrects)
top_wrong = np.flip(top_wrong)
if n_incorrects<10:
    top_n = top_wrong[0:n_incorrects]
else:
    top_n = top_wrong[0:10]


#Plotting wrongly classified instances and saving them if needed
for i in range(np.size(top_n)):
    plt.imshow(x_test[top_n[i]], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)    
    plt.xlabel('Correct: '+class_names[y_test_[top_n[i]]]+' / Predicted:'+class_names[y_predicted[top_n[i]]])
    #plt.savefig(path_2+'/'+str(i+1)+'_Model2_Correct'+class_names[y_test_[top_n[i]]]+'_Predicted'+class_names[y_predicted[top_n[i]]] +'.jpg' )
    plt.show()

##### Model 3 #####
# Predicting probability outputs for test data
y_prob = model3.predict(x_test)

# Obtaining the highest probabilities, the 'most likely' to be correct class
y_predicted = y_prob.argmax(axis=-1)

#Obtaining the correctly and incorrectly classified class vs the ground truth
y_test_=y_test_.reshape((-1,))
incorrects = np.nonzero(y_predicted != y_test_)
corrects = np.nonzero(y_predicted == y_test_)
y_prob[corrects] = 0
prob_incorrects= (np.max(y_prob,axis=1))

#Obtain the number of incorrectly classified instances
n_incorrects  = np.sum(prob_incorrects!=0)
print('Number of incorrecty classified numbers on test set, for model 3: ' + str(n_incorrects))

#Get the worst 10 classified instances
top_wrong = np.argsort(prob_incorrects)
top_wrong = np.flip(top_wrong)
if n_incorrects<10:
    top_n = top_wrong[0:n_incorrects]
else:
    top_n = top_wrong[0:10]


#Plotting wrongly classified instances and saving them if needed
for i in range(np.size(top_n)):
    plt.imshow(x_test[top_n[i]], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)    
    plt.xlabel('Correct: '+class_names[y_test_[top_n[i]]]+' / Predicted:'+class_names[y_predicted[top_n[i]]])
    #plt.savefig(path_3+'/'+str(i+1)+'_Model3_Correct'+class_names[y_test_[top_n[i]]]+'_Predicted'+class_names[y_predicted[top_n[i]]] +'.jpg' )
    plt.show()