from __future__ import print_function

#import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

#print('tensorflow:', tf.__version__)
#print('tensorflow.keras:', tensorflow.keras.__version__)


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


#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = tensorflow.keras.utils.to_categorical(y_train)
y_test = tensorflow.keras.utils.to_categorical(y_test)


num_classes = y_train.shape[1]


# =============================================================
# For different optimizers functions
# =============================================================
optimizers = ['SGD', 'Adam', 'RMSprop', 'Adamax', 'Ftrl']

#Initialize list of empty accuracies and losses
val_accs = []
val_losses = []

#Iterate over list of optimizers
for counter, optimizer in enumerate(optimizers):
    #Define Model of fully connected layers
    model = Sequential()
    model.add(Dense(64 ,activation = 'sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    #Compile the model given the different optimizers
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #Fit model for each different optimizer
    history = model.fit(x_train,
                            y_train,
                            epochs = 100,
                            batch_size = 1024,
                            validation_split=0.33)
    #Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))
    
    #Fetch training and validation losses
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    val_losses.append(val_loss)
    
    #Plot loss graphs for each optimizer
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.title('validation losses')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    #plt.savefig('val_loss_' +  str(optimizer) + '.png')
    #plt.close()
    plt.show()
    
    #Plot accuracy graphs for each optimizer
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    val_accs.append(val_acc)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['acc', 'val_acc'])
    plt.title('accuracies')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    #plt.savefig('accuracy_loss_' + str(optimizer) + '.png')
    #plt.close()
    plt.show()

#Plot validating graphs for different optimizers
plt.plot(val_accs[0])
plt.plot(val_accs[1])
plt.plot(val_accs[2])
plt.plot(val_accs[3])
plt.plot(val_accs[4])
plt.legend(['SGD', 'Adam', 'RMSprop', 'Adamax', 'Ftrl'])
plt.title('accuracies')
plt.xlabel('iteration')
plt.ylabel('accuracy')
#plt.savefig('accuracy_loss_neurons.png')
#plt.close()
plt.show()

#Plot validating losses for different optimizers
plt.plot(val_losses[0])
plt.plot(val_losses[1])
plt.plot(val_losses[2])
plt.plot(val_losses[3])
plt.plot(val_losses[4])
plt.legend(['SGD', 'Adam', 'RMSprop', 'Adamax', 'Ftrl'])
plt.title('losses')
plt.xlabel('iteration')
plt.ylabel('loss')
#plt.savefig('validation_loss_neurons.png')
plt.show()