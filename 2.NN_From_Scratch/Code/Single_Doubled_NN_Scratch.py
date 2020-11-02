from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


# Importing dataset, splitting it into training and testing
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Preparing the data for Training and Testing
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255


# Adjust outputs for binary classification: digit 0 is classified 1 
# and all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


y_train = y_train.T
y_test = y_test.T

# Number of training instances
m = X_train.shape[1] 

# Shuffle training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]

# =============================================================================
# For Report Purposes
# 
# # Display one image and corresponding label
# import matplotlib
# import matplotlib.pyplot as plt
# i = 3
# print('y[{}]={}'.format(i, y_train[:,i]))
# plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()
# =============================================================================




#Creating a Neural Network

# Define Sigmoid Activation Function
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

# For a single Neuron
def singleNeuron(x, W, b, sigmoid):
    return sigmoid(np.dot(W, x) + b)

# Define a Binary loss function 
def loss_function(y, y_hat):
    m = y.shape[1]
    return -1/m * ( np.sum(np.multiply(np.log(y_hat),y)) + np.sum(np.multiply(np.log(1-y_hat),(1-y))) )

# Backpropagate a neuron
def backpropagation_single_neuron(y, y_hat, w, X):
    m = y.shape[1]
    dX = y_hat - y
    dW = np.dot(dX,X.T)/ m
    db = np.sum(dX) / m
    return dW, db, dX

# Weight and Bias Update 
def update_single_neuron(W, dW, b, db, h):
    W = W - h * dW
    b = b - h * db
    return W,b

# Define the training process for a single layer, single neuron neetowrk
def train(x, y, epochs = 5, h = 0.9):
    print("\n\n==========START TRAINING FOR SINGLE LAYER ==========\n\n")
    layer_output_size = 1
    np.random.seed(22)
    W = np.random.randn(layer_output_size, 784) * 1/100
    b = np.random.randn(layer_output_size,1) * 1/100

    train_losses = []
    train_accuracies = []

    for i in range(epochs):
        print("Starting epoch " + str(i))
        y_hat = singleNeuron(x, W, b, sigmoid)
        
        loss = loss_function(y, y_hat)
        print("Train Loss: " +str(np.round(loss,4)))
        train_losses.append(loss)
        
        accur = accuracy(y_hat,y)
        print("Train accuracy: " + str(np.round(accur,4))+"\n")
        train_accuracies.append(accur)
        
        dW, db, dX = backpropagation_single_neuron(y, y_hat, W, x)
        W,b = update_single_neuron(W, dW, b, db, h)


    return train_losses, train_accuracies, W, b

# Define accuracy based on binary classification. Simple threshold
def accuracy(y, y_hat):
    y_hat_corrected = y_hat > 0.5
    errors = np.sum(np.abs(y-y_hat_corrected))
    return 1 - errors / y.shape[1]

# Define a Sequential two layered network
def two_layers_network(x, W_h, b_h, W_o, b_o):
    z = singleNeuron(x, W_h, b_h, sigmoid)
    return singleNeuron(z, W_o, b_o, sigmoid)


# Define Training Method for a Two Layered Network
def train_two_layers(x, y, X_test, epochs = 5, h = 0.9):
    print("\n\n==========START TRAINING FOR DOUBLE LAYER ==========\n\n")
    np.random.seed(22)
    W_h = np.random.randn(64, 784) * 1/100
    b_h = np.random.randn(64,1) * 1/100
    W_o = np.random.randn(1, 64) * 1/100
    b_o =np.random.randn(1,1) * 1/100

    train_losses = []
    train_accuracies = []
    for i in range(epochs):
        
        print("Starting epoch " + str(i))
        z = singleNeuron(x, W_h, b_h, sigmoid)
        y_hat = singleNeuron(z, W_o, b_o, sigmoid)
        
        loss = loss_function(y, y_hat)
        print("Train Loss: " +str(np.round(loss,4)))
        train_losses.append(loss)
        
        accur = accuracy(y_hat,y)
        print("Train accuracy: " + str(np.round(accur,4))+"\n")
        train_accuracies.append(accur)
        
        dW_o, db_o, dW_h, db_h = backpropagation_two_layers(y, y_hat, W_h, W_o, x, z)
        W_o, b_o, W_h, b_h = update_two_layers(dW_o, db_o, dW_h, db_h, W_h, b_h, W_o, b_o, h)

    return train_losses, train_accuracies, W_o, b_o, W_h, b_h

# Backpropagate on both layers, neurons
def backpropagation_two_layers(y, y_hat, W_h, W_o, x, z):
    dW_o, db_o, dX_o = backpropagation_single_neuron(y, y_hat, W_o, z)
    m = y.shape[1]
    dX1 = np.dot(W_o.T,dX_o)
    dX1 = np.multiply(np.multiply(dX1, z),1 - z)
    dW_h = np.dot(dX1, x.T) / m
    db_h = np.sum(dX1) / m
    return dW_o, db_o, dW_h, db_h

# Weights and Bias Updates for double layered network
def update_two_layers(dW_o, db_o, dW_h, db_h, W_h, b_h, W_o, b_o, h):
    W_o = W_o - h * dW_o
    b_o = b_o - h * db_o
    W_h = W_h - h * dW_h
    b_h = b_h - h * db_h

    return W_o, b_o, W_h, b_h


epochs=20
#Single Layer
train_losses, train_accuracies, W, b = train(X_train, y_train, epochs)
y_hat_test = singleNeuron(X_test, W, b, sigmoid)
test_loss = loss_function(y_test, y_hat_test)
print("Test Accuracy: " + str(accuracy(y_test, y_hat_test)))

#Double Layer
train_losses_2, train_accuracies_2, W_o, b_o, W_h, b_h = train_two_layers(X_train, y_train, X_test, epochs)
y_hat_test = two_layers_network(X_test, W_h, b_h, W_o, b_o)
test_loss = loss_function(y_test, y_hat_test)
print("Test Accuracy: " + str(accuracy(y_test, y_hat_test)))

plt.figure()
plt.plot(train_losses)
plt.plot(train_losses_2)
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend(['Single Layer', 'Double Layer'])
plt.show()

plt.figure()
plt.plot(train_accuracies)
plt.plot(train_accuracies_2)
plt.title("Training Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend(['Single Layer', 'Double Layer'])
plt.show()