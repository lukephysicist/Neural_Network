import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class Layer:
    
    def __init__(self, n_inputs, n_neurons, activation_func):

        if activation_func == 'sigmoid':
            self.activation_func = sigmoid

        elif activation_func == 'softmax':
            self.activation_func = softmax
        
        elif activation_func == 'relu':
            self.activation_func = relu

        else: self.activation_func = linear

        if self.activation_func == relu:
            self.weights = np.sqrt(2/n_inputs) * np.random.randn(n_inputs, n_neurons)
        else:
            self.weights = np.sqrt(1/n_inputs) * np.random.randn(n_inputs, n_neurons)

        self.biases = np.zeros((1, n_neurons))
        self.next_layer = None

    
    def forward(self, inputs):
        self.a_prev = inputs
        self.z = np.dot(self.a_prev, self.weights) + self.biases
        self.a = self.activation_func(self.z)
        return self.a
    
    def back(self, truths, rate, loss_func):
        if loss_func == mean_sq_error:
            # change in cost w.r.t. activation in the current layer
            self.delC_delA = self.calc_delC_delA(truths)

            # change in activation w.r.t. weighted sum in current layer
            self.delA_delZ = act_prime(self.activation_func, self.z)

            # change in weighted sum w.r.t. inputs to the current layer
            self.delZ_delW = self.a_prev.T

            # change in cost w.r.t. weights in the current layer
            delC_delW = np.dot(self.delZ_delW, self.delA_delZ * self.delC_delA) / self.a.shape[0]

            self.weights -= (delC_delW * rate) 


            # change in cost w.r.t. biases in the current layer
            delC_delB = np.mean(self.delC_delA * self.delA_delZ, axis = 0)

            self.biases -= delC_delB * rate

    def calc_delC_delA(self, truths):
        if self.next_layer == None:
            return 2 * (self.a - truths)
        
        next_delC_delA = self.next_layer.delC_delA
        next_delA_delZ = self.next_layer.delA_delZ
        next_delZ_delA_prev = self.next_layer.weights.T

        return np.dot(next_delC_delA * next_delA_delZ, next_delZ_delA_prev)
        


class Network:
    
    def __init__(self, layers, rate = .01, regressor = True):
        if not all(isinstance(layer, Layer) for layer in layers):
            raise NetworkErrorOne("""must initialize Network with list of Layer objects.""")
        
        
        for i in range(len(layers) - 1):
            if layers[i].weights.shape[1] != layers[i+1].weights.shape[0]:
                raise NetworkErrorTwo("Each layer must have n_inputs equal to n_neurons in previous layer")
            else:
                layers[i].next_layer = layers[i+1]

        layers[-1].next_layer = None
        self.layers = layers
        self.rate = rate
        self.loss = None
        self.regressor = regressor
        self.loss_func = mean_sq_error if self.regressor else cat_cross_entropy


    def network_forward(self, network_inputs, truths = np.array([])):

        if network_inputs.shape[1] != self.layers[0].weights.shape[0]:
            raise NetworkErrorThree(f'''
                the network has {self.layers[0].weights.shape[0]} dimensional inputs.
                You gave a {network_inputs.shape[1]} dimensional input'''
                )

        layer_inputs = network_inputs
        for layer in self.layers:
            layer_inputs = layer.forward(layer_inputs)

        self.out = layer_inputs
        self.loss = self.loss_func(self.out, truths)

        return self.out

    def network_back(self, truths):
        for layer in reversed(self.layers):
            layer.back(truths, self.rate, self.loss_func)

    def train_test(self, training_samples, testing_samples, batch_size, n_epochs):
        
        def new_epoch(traning_samples, batch_size):
            # shuffles the traning data 
            random.shuffle(list(training_samples.items()))
            traning_samples = list(traning_samples)

            training_set = [dict(list(training_samples.items())[i: i+batch_size]) for i in range(0, len(training_samples.items()), batch_size)]

            network_inputs = [np.array(list(batch.keys())) for batch in training_set]
            network_truths = [np.array(list(batch.values())).reshape(-1,1) for batch in training_set] # <---- reshape may have to be different for classifier

            return (network_inputs, network_truths)
        
        # trains the network on each batch in the traning set
        loss_ys = []
        for j in range(n_epochs):
            # shuffle the batches for every epoch
            network_inputs, network_truths = new_epoch(training_samples, batch_size)
            for i in range(len(network_inputs)):
                self.network_forward(network_inputs=network_inputs[i], truths=network_truths[i])
                loss_ys.append(self.loss)
                self.network_back(network_truths[i])
            


        # creates a list of testing inputs batches and their answers
        testing_inputs = np.array(list(testing_samples.keys()))
        testing_truths = np.array(list(testing_samples.values())).reshape(-1,1)

        testing_preds = self.network_forward(testing_inputs, testing_truths)
        loss_final = self.loss
        loss_ys.append(loss_final)

        if self.regressor:
            # plot loss after each forward pass
            plt.plot(loss_ys)
            plt.xlabel('Batch')
            plt.ylabel('Mean Sq Error')
            plt.title('Loss')
            plt.show()
            print(f'MSE: {loss_final}\n')

            df = pd.DataFrame(
                {
                "Model Predictions" : [pred[0] for pred in testing_preds],
                "Truths" : [truth[0] for truth in testing_truths]
                }
            )
            return (df)


    

class NetworkErrorOne(Exception):
    """For if the network object is not passed a list"""
    


class NetworkErrorTwo(Exception):
    """for if adjacent layers dont fit together"""


class NetworkErrorThree(Exception):
    """for when an incorrect number of inputs is provided to the network"""

def relu(inputs):
    return np.maximum(0, inputs)
    
def sigmoid(inputs):
    inputs = np.clip(inputs, -5, 5)
    return 1 / (1 + np.exp(-inputs))

def softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
def linear(x):
    return x

def act_prime(func, x):
    if func == sigmoid:
        return sigmoid_prime(x)
    elif func == relu:
        return relu_prime(x)
    elif func == linear:
        return np.ones(x.shape)
    
def relu_prime(x):
    return np.where(x > 0, 1, 0)

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1-s)
    

def cat_cross_entropy(inputs, truths):
    batch_size = len(inputs)
    clipped_inputs = np.clip(inputs, 1e-7, 1-1e-7)

    if len(truths.shape) == 1:
        true_estimates = clipped_inputs[range(batch_size), truths]
    else:
        true_estimates = np.sum(clipped_inputs * truths, axis=1)
    
    logged_true_estimates = -np.log(true_estimates)

    return np.mean(logged_true_estimates)

def mean_sq_error(inputs, truths):

    error_matrix = np.square(inputs - truths)
    sum_vector = np.sum(error_matrix, axis = 1)

    return np.mean(sum_vector)