from __future__ import print_function
#import tensorflow as tf
#import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import LeakyReLU
#print('tensorflow:', tf.__version__)
#print('keras:', tensorflow.keras.__version__)


#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8


#To input our values in our network Dense layer, we need to flatten the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 784)
#flatten images
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255


#For binary classification 1 vs all approach.
y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new


y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


num_classes = 1




# =============================================================
# For different batch sizes
# =============================================================
batch_sizes = [32768, 512, 1024, 8192]

#Initialize list of empty accuracies and losses
val_accs = []
val_losses = []

#Keep track of time for analysis
times = []

#Iterating over every batch size
for counter, batch_size in enumerate(batch_sizes):
    
    #Start stopwatch
    start = time.time()
    
    #Define Sequential model
    model = Sequential()
    #Add a Fully connected layer with sigmoid activation function
    model.add(Dense(1 ,activation = 'sigmoid'))
    #Compile the model for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #Fit the model given the different batch sizes
    history = model.fit(x_train,
                        y_train,
                        epochs = 100,
                        batch_size = batch_size,
                        validation_split=0.33)
    
    #Finish stopwatch
    end = time.time()
    #Display elapsed time
    print('THE TIME: ' + str(end-start))
    
    times.append(end - start)
    
    #Evaluate Model
    _, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))
    
    #Fetch training and validation losses
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    val_losses.append(val_loss)
    
    #Plot loss graphs for each batch size
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.title('validation losses')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    #plt.savefig('val_loss_' +  str(batch_size) + '.png')
    plt.show()

    #Plot accuracy graphs for earch batch size
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    val_accs.append(val_acc)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['acc', 'val_acc'])
    plt.title('accuracies')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    #plt.savefig('accuracy_loss_' + str(batch_size) + '.png')
    plt.show()

print(times)

#Plot accuracy graphs for different batch sizes
plt.plot(val_accs[0])
plt.plot(val_accs[1])
plt.plot(val_accs[2])
plt.plot(val_accs[3])
plt.legend(['batch size: 32,768', 'batch size: 512', 'batch size: 1,024','batch size: 8,192'])
plt.title('accuracies')
plt.xlabel('iteration')
plt.ylabel('accuracy')
#plt.savefig('accuracy_loss_batch_sizes.png')
plt.show()

#Plot loss graphs for different batch sizes
plt.plot(val_losses[0])
plt.plot(val_losses[1])
plt.plot(val_losses[2])
plt.plot(val_losses[3])
plt.legend(['batch size: 32,768', 'batch size: 512', 'batch size: 1,024','batch size: 8,192'])
plt.title('losses')
plt.xlabel('iteration')
plt.ylabel('loss')
#plt.savefig('validation_loss_batch_sizes.png')
plt.show()


# =============================================================
# For different activation functions
# =============================================================

activations = ['LeakyRelu', None, 'tanh', 'exponential', 'relu']

#Initialize list of empty accuracies and losses
val_accs = []
val_losses = []

for counter, activation_ in enumerate(activations):
    #Build Sequential model
    model = Sequential()    
    if activation_=='LeakyRelu':
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.05))
    else:
        model.add(Dense(64, activation_ ))
        model.add(Dense(1, activation_))
    
    #Compile the model 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #Fit the model given the different activation functions
    history = model.fit(x_train,
                        y_train,
                        epochs = 100,
                        batch_size = 1024,
                        validation_split=0.33)
    
    
    _, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))
    
    #Fetch training and validation losses
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    val_losses.append(val_loss)
    
    #Plot loss graphs for each activation function
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.title('validation losses')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    #plt.savefig('val_loss_' +  str(activation_) + '.png')
    #plt.close()
    plt.show()

    #Plot accuracy graphs for each activation function
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    val_accs.append(val_acc)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['acc', 'val_acc'])
    plt.title('accuracies')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    #plt.savefig('accuracy_loss_' + str(activation_) + '.png')
    #plt.close()
    plt.show()


# Plotting validation accuracy results for different activation functions
activations = ['LeakyReLU', None, 'tanh', 'exponential', 'relu']
plt.plot(val_accs[0])
plt.plot(val_accs[1])
plt.plot(val_accs[2])
plt.plot(val_accs[3])
plt.plot(val_accs[4])
plt.legend(['LeakyReLU', 'None', 'tanh', 'exponential', 'relu'])
plt.title('accuracies')
plt.xlabel('iteration')
plt.ylabel('accuracy')
#plt.savefig('accuracy_loss_neurons.png')
#plt.close()
plt.show()

# Plotting validation losses results for different activation functions
plt.plot(val_losses[0])
plt.plot(val_losses[1])
plt.plot(val_losses[2])
plt.plot(val_losses[3])
plt.plot(val_losses[4])
plt.legend(['LeakyReLU', 'None', 'tanh', 'exponential', 'relu'])
plt.title('losses')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('validation_loss_neurons.png')
plt.show()
