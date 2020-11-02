from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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




y_train = y_train.T
y_test = y_test.T

# Number of training instances
m = X_train.shape[1] 

#Shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]

"""
# Adjust outputs for binary classification: digit 0 is classified 1 
# and all the other digits are classified 0
y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new
"""


class Network:
    def __init__(self, network_architecture, seed = 22):
        '''
        Initialize the network
        :param network_architecture: the architecture
        :param seed: to be able to reproduce the results
        '''
        np.random.seed(seed)
        self.init_layers(network_architecture)


    def init_layers(self, network_architecture):
        '''
        Initialize the layers based on the network architecture, and fill them with variables sampled from normal distribution
        :param network_architecture: the architecture
        '''
        self.weights = [] #the list of the weights of the layers
        self.b = [] #the list of biases
        self.activation = [] #the activation function for each layer
        self.architecture = network_architecture
        for i in range(len(network_architecture)):
            layer = network_architecture[i]
            layer_input_size = layer["input"]
            layer_output_size = layer["output"]

            #Initialize with random values
            W = np.random.randn(layer_output_size, layer_input_size) * 1/100
            b = np.random.randn(layer_output_size,1) * 1/100

            #Save the weights and biases, layer 0 will be the first layer, the input is saved separately
            self.weights.append(W)
            self.b.append(b)

            #Set up the activation function, and define what loss will be use at the evaluation (only the last layers accuracy will matter)
            if layer["activation"] == 'softmax':
                self.activation.append(self.softmax)
                self.accuracy = self.accuracy_multiclass
            else:
                self.activation.append(self.sigmoid)
                self.accuracy = self.accuracy_binary
                
    # Define Binary Activation Functoin
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    # Define Multiclass Activation Function
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z)) #add some constant to avoid the division by zero
        return expZ / expZ.sum(axis=0, keepdims=True)

    def singleLayer(self, X, layer_idx):
        '''
        Forward propagation of X on a single layer, gives by layer index
        :param X: the input of the layer
        :param layer_idx: the index of the layer (with the index will be able to reach the weights, bias and activation function belonging to that layer
        :return: the output of the layer after the activaion
        '''

        #calculate and save the output of the layer before the activation
        Z = np.dot(self.weights[layer_idx], X) + self.b[layer_idx]
        self.Z.append(Z)
        #apply the activation function
        activation_function = self.activation[layer_idx]
        result = activation_function(Z)
        return result

    def feed_forward_iteration(self, X):
        '''
        Propagate the input through the network
        :param X: input of the network
        :return: the output of the network
        '''
        self.input = X
        current_output = X
        self.X = [] #the output of each layer with the activation function
        self.Z = [] #the output of each layer before activation
        for layer_idx in range(len(self.architecture)):
            previous_output = current_output
            current_output = self.singleLayer(previous_output, layer_idx) #calculate the output of the layer
            self.X.append(current_output)

        return current_output

    def predict(self, X):
        '''
        Prediction from data X, using the defined network
        :param X: input data
        :return: prediction
        '''
        current_output = X
        for layer_idx in range(len(self.architecture)):
            previous_output = current_output
            current_output = self.singleLayer(previous_output, layer_idx)
        return current_output

    def loss(self, Y_predicted, Y_true):
        '''
        Calculate the cross entropy in binary prediction
        :param Y_predicted: predicitons
        :param Y_true: true values
        :return:
        '''
        m = Y_predicted.shape[1]
        return -1/m * ( np.sum(np.multiply(np.log(Y_predicted),Y_true)) + np.sum(np.multiply(np.log(1-Y_predicted),(1-Y_true))) )

    def loss_multiclass(self, Y_predicted, Y_true):
        '''
        Calculate the cross entropy loss in multiclass prediction
        :param Y_predicted: predictions
        :param Y_true: true values
        :return:
        '''
        return -np.mean(Y_true * np.log(Y_predicted + 1e-8)) #small value added to avoid the division by zero

    def sigmoid_backward(self, dA, Z):
        '''
        Derivate of Sigmoid function used for backpropagation
        :param dA: loss the backpropagate
        :param Z: output of the layer (before the activation applied)
        :return:
        '''
        sig = self.sigmoid(Z)
        return dA * sig * (1-sig)


    def backpropagate_single_layer(self, dX, Z, X_prev, W, m):
        '''

        :param dX: loss to backpropagate
        :param Z: output of layer before activation
        :param X_prev: output of pervious layer
        :param W: weights
        :param m: number of samples
        :return: values to update weights, bias and the loss for further backpropagation
        '''
        dZ = self.sigmoid_backward(dX, Z)
        dW = np.dot(dZ, X_prev.T) / m
        db = np.sum(dZ) / m
        dX_prev = np.dot(W.T, dZ)

        return dW, db, dX_prev

    def backward_propagation(self, Y_true):
        '''
        Backpropagation for the whole network
        :param Y_true: true values of y
        :return:
        '''
        m = Y_true.shape[-1]
        #Backpropagation for the head of the network
        dX = self.X[len(self.architecture)-1] - Y_true
        self.dW = [None] * len(self.weights)
        self.db = [None] * len(self.b)
        last_layer_idx = len(self.architecture)-1
        if last_layer_idx > 0:
            X_prev = self.X[last_layer_idx-1]
        else:
            X_prev = self.input
        self.dW[last_layer_idx] = np.dot(dX, X_prev.T) / m
        self.db[last_layer_idx] = np.sum(dX) / m
        dX = np.dot(self.weights[last_layer_idx].T, dX)
        #Backpropagation for the body
        for layer_idx in range(len(self.architecture)-2,-1,-1):
            if layer_idx > 0: #Check if we reach the bottom of the network (in case the next layer is the input)
                X_prev = self.X[layer_idx-1]
            else:
                X_prev = self.input
            _dW, _db, dX_prev = self.backpropagate_single_layer(dX, self.Z[layer_idx], X_prev, self.weights[layer_idx], m) #backpropagate thorugh each layer
            self.dW[layer_idx] = _dW
            self.db[layer_idx] = _db

            dX = dX_prev

    def update(self):
        '''
        Update the weights and biases
        '''
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.learning_rate * self.dW[i]
            self.b[i] = self.b[i] - self.learning_rate * self.db[i]


    def train(self, X, Y, batch_size, epochs, learning_rate=0.9):
        '''
        training of the network
        :param X: input
        :param Y: true values
        :param batch_size: size of batch for each iteration
        :param epochs: number of epochs
        :param learning_rate: validation loss and accuracy
        :return:
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        train_losses = []
        val_losses = []
        accuracies = []
        X_train, X_val, Y_train, Y_val = train_test_split(X.T, Y.T, test_size = 0.33, random_state = 42)
        X_train, X_val, Y_train, Y_val = X_train.T, X_val.T, Y_train.T, Y_val.T

        for epoch in range(self.epochs):
            print('Epoch ' + str(epoch))
            for index in range(0,X_train.shape[1],self.batch_size):
                x = X_train[:, index:min(index + self.batch_size, X_train.shape[1])]
                y = Y_train[:,index:min(index + self.batch_size, X_train.shape[1])]

                x_val = X_val
                y_val = Y_val

                output = self.feed_forward_iteration(x)
                train_loss = self.loss_multiclass(output, y)
                train_losses.append(train_loss)

                y_prediction = self.predict(x_val)
                val_loss = self.loss_multiclass(y_prediction, y_val)

                #print(str(min(index + self.batch_size, X_train.shape[1])) + '/' + str(X_train.shape[1]) + ' train loss: ' + str(train_loss) + ' val loss: ' + str(val_loss))
                self.backward_propagation(y)
                self.update()

            Y_prediction = self.predict(X_val)
            val_loss = self.loss_multiclass(Y_prediction, Y_val)
            val_losses.append(val_loss)

            acc = self.accuracy(Y_val, Y_prediction)
            print('accuracy: ' + str(acc))
            accuracies.append(acc)


        return val_losses, accuracies

    def accuracy_binary(self, y, y_hat):
        y_hat_corrected = y_hat > 0.5
        errors = np.sum(np.abs(y - y_hat_corrected))
        return 1 - errors / y.shape[1]

    def accuracy_multiclass(self, Y_val, Y_prediction):
        y_hat_discrete = np.argmax(Y_prediction, axis=0)
        y_discrete = np.argmax(Y_val, axis=0)
        accuracy = 0
        for i in range(len(y_discrete)):
            if y_discrete[i] == y_hat_discrete[i]:
                accuracy += 1
        acc = accuracy/y_discrete.shape[0]
        return acc


def preprocess_data(y):
    y = y.astype('int')
    b = np.zeros((y.size, y.max() + 1))
    b[np.arange(y.size), y] = 1
    return b.T

"""
network_architecture = [
    {'input': 784, 'output': 1, 'activation': 'sigmoid'}
]


network_architecture = [
    {'input': 784, 'output': 64, 'activation': 'sigmoid'},
    {'input': 64, 'output': 1, 'activation': 'sigmoid'},
]


neural_network = Network(network_architecture) #define the network
val_losses, accr = neural_network.train(X_train,
                                    y_train,
                                    batch_size = 8192,
                                    epochs=10,
                                    learning_rate=0.9)

y_predict = neural_network.predict(X_test) #prediction for the test, based on the learned weights
acc = neural_network.accuracy_binary(y_test,y_predict) # binary accuracy

print('Test accuracy: ' + str(acc))
plt.plot(val_losses)
plt.title('Validation loss')
#plt.legend(('validation loss'))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.plot(accr)
plt.title('Accuracy')
plt.show()

"""

# Define the network architecture
network_architecture = [
    {'input': 784, 'output': 64, 'activation': 'sigmoid'},
    {'input': 64, 'output': 10, 'activation': 'softmax'},
]

neural_network = Network(network_architecture) #define the network
prep_y = preprocess_data(y_train) #in case of multiclass we need to transform the data
val_losses, accr = neural_network.train(X_train,
                                    prep_y,
                                    batch_size = 8192,
                                    epochs=500,
                                    learning_rate=0.01)

y_predict = neural_network.predict(X_test) #prediction for the test, based on the learned weights
acc = neural_network.accuracy_multiclass(preprocess_data(y_test),y_predict) # multiclass accuracy

print('Test accuracy: ' + str(acc))

plt.plot(val_losses)
plt.title('Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.plot(accr)
plt.title('Accuracy')
plt.show()
