from __future__ import print_function

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
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


#load (first download if necessary) the CIFAR10 dataset
# data is already split in train and test datasets
(x_train, y_train_), (x_test, y_test_) = cifar10.load_data()

#Convert to float
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')


# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

num_classes = 10

#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = tensorflow.keras.utils.to_categorical(y_train_, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test_, num_classes)

#For Model Lab 3.3
num_pixels = x_train.shape[1] * x_train.shape[2]*x_train.shape[3]
x_train_3  = x_train.reshape(x_train.shape[0], num_pixels)
x_test_3   = x_test.reshape(x_test.shape[0], num_pixels)

#Let start our work: creating a convolutional neural network

# =============================================================================
# For the report
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']
# 
# plt.figure(figsize=(5,2))
# for i in range(10):
#     plt.subplot(5,2,i+1)
#     plt.imshow(x_test[i], cmap=plt.cm.binary)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)    
#     plt.xlabel(class_names[y_test[i][0]])
#     #plt.savefig(class_names[y_test[i][0]] + '.jpg')
#     plt.show()
# 
# =============================================================================

_epochs=3;
# =============================================================================
# Defining Models
# =============================================================================
# ##### Model 1 #####

# Define Model Type
model1 = Sequential()

# Add single convolutional layer 
model1.add(Conv2D(64,
                 kernel_size = (3, 3),
                 activation = 'relu',
                 strides = (1,1)))

# Flatten into one dimension feature vector per image sample
model1.add(Flatten())

# Add a fully connected layer
model1.add(Dense(num_classes, activation='softmax'))

# Compile the model with best optimizer
model1.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model for 50 epochs
history1 = model1.fit( x_train,
                            y_train,
                            epochs = _epochs,
                            batch_size = 1024,
                            validation_split=0.2)

# =============================================================================
# #### Lab 3.3 model ####
# Define model type
model_lab3 = Sequential()

# Add a single hidden layer with the best activation function found in lab 3
model_lab3.add(Dense(64 ,activation = 'relu'))

# Add last fully connected layer with categorical activation function
model_lab3.add(Dense(num_classes, activation='softmax'))

# Compile the model given the best optimizer
model_lab3.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model for 50 epochs given the optimal batch size of 1024
history_lab3 = model_lab3.fit(x_train_3, y_train, epochs = _epochs, batch_size = 1024, validation_split=0.2)

# =============================================================================
# Evaluate both models on test set
testloss_1, testaccuracy_1       = model1.evaluate(x_test, y_test)
testloss_lab3, testaccuracy_lab3 = model_lab3.evaluate(x_test_3, y_test)


# Fetching the losses for both models to be compared
loss1         = history1.history['loss']
val_loss1     = history1.history['val_loss']
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
acc1         = history1.history['accuracy']
val_acc1     = history1.history['val_accuracy']
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
 
# =============================================================================
# #### Model 2 ####

# Define Model Type
model2 = Sequential()

# Add first convolutional layer
model2.add(Conv2D(64,
                 kernel_size = (3, 3),
                 activation = 'relu',
                 strides = (1,1)))

# Max Pooling between convolutional layers
model2.add(MaxPooling2D((3,3), strides = (1,1), padding = 'same'))

# Add second convolutional layer
model2.add(Conv2D(128,
                 kernel_size = (3, 3),
                 activation = 'relu',
                 strides = (1,1)))

# Flatten into one dimension feature vector per image sample
model2.add(Flatten())

# Add a fully connected layer
model2.add(Dense(num_classes, activation='softmax'))

# Compile the model with best optimizer
model2.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model for 50 epochs
history2 = model2.fit(x_train,
                            y_train,
                            epochs = _epochs,
                            batch_size = 1024,
                            validation_split=0.2)
# =============================================================================
# #### Model 3 ####

# Define Model Type
model3 = Sequential()

# Add first convolutional layer
model3.add(Conv2D(64,
                 kernel_size = (3, 3),
                 activation = 'relu',
                 strides = (1,1)))

# Max Pooling between convolutional layers
model3.add(MaxPooling2D((3,3), strides = (1,1), padding = 'same'))

# Add second convolutional layer
model3.add(Conv2D(128,
                 kernel_size = (3, 3),
                 activation = 'relu',
                 strides = (1,1)))

# Max Pooling between convolutional layers
model3.add(MaxPooling2D((3,3), strides = (1,1), padding = 'same'))

# Add third convolutional layer
model3.add(Conv2D(256,
                 kernel_size = (3, 3),
                 activation = 'relu',
                 strides = (1,1)))

# Flatten into one dimension feature vector per image sample
model3.add(Flatten())

# Add a fully connected layer
model3.add(Dense(num_classes, activation='softmax'))

# Compile the model with best optimizer
model3.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model for 50 epochs
history3 = model3.fit(x_train,
                            y_train,
                            epochs = _epochs,
                            batch_size = 1024,
                            validation_split=0.2)
# =============================================================================
# EValuate three convolutional neural networks on test set
testloss_1, testaccuracy_1 = model1.evaluate(x_test, y_test)
testloss_2, testaccuracy_2 = model2.evaluate(x_test, y_test)
testloss_3, testaccuracy_3 = model3.evaluate(x_test, y_test)

# Fetching the losses for all three models to be compared
loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']
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
acc1 = history1.history['accuracy']
val_acc1 = history1.history['val_accuracy']
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

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

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

# =============================================================================
