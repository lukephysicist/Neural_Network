import numpy as np

class DenseLayer:
    
    def __init__(self, n_inputs, n_neurons, activation_func):
        self.weights = .1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        if activation_func == 'sigmoid':
            self.activation_func = sigmoid

        elif activation_func == 'softmax':
            self.activation_func = softmax

        else: self.activation_func = relu
    
    def forward(self, inputs):
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_func(self.z)
        return self.a
    
    def back(self, inputs, truths, rate):

        # assumes loss is MSE:
        e_wise_product = act_prime(self.activation_func, self.z) * 2 * (self.a - truths)

        weight_deltas = np.dot(inputs.T, e_wise_product) / inputs.shape[0]
        self.weights -= (weight_deltas * rate) 

        bias_deltas = np.mean(act_prime(self.activation_func, self.z) * 2 * (self.a - truths), axis = 0)
        self.biases -= bias_deltas * rate



class Network:
    
    def __init__(self, layers, loss_func = 'mse'):
        if not all(isinstance(layer, DenseLayer) for layer in layers):
            raise NetworkErrorOne("""must initialize Network with list of Layer objects.""")
        
        
        for i in range(len(layers) - 1):
            if layers[i].weights.shape[1] != layers[i+1].weights.shape[0]:
                raise NetworkErrorTwo("Each layer must have n_inputs equal to n_neurons in previous layer")
                
        self.layers = layers

        if loss_func == 'cat_cross_entropy':
            self.loss_func = cat_cross_entropy
        
        else: self.loss_func = mean_sq_error


    def network_forward(self, network_inputs, truths = np.array([])):
        if network_inputs.shape[0] != self.layers[0].weights.shape[0]:
            raise NetworkErrorThree(f'''
                the network has {self.layers[0].weights.shape[0]} dimensional inputs.
                You gave a{network_inputs.shape[0]} dimensional input''')

        layer_inputs = network_inputs
        for layer in self.layers:
            layer_inputs = layer.forward(layer_inputs)

        self.out = layer_inputs

        if len(truths) != 0:
            self.loss = self.loss_func(self.out, truths)

        return self.out



class NetworkErrorOne(Exception):
    """For if the network object is not passed a list"""
    


class NetworkErrorTwo(Exception):
    """for if adjacent layers dont fit together"""


class NetworkErrorThree(Exception):
    """for when an incorrect number of inputs is provided to the network"""

def relu(inputs):
    return np.maximum(0, inputs)
    
def sigmoid(inputs):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        return sigmoid(inputs)

def act_prime(func, x):
    if func == sigmoid:
        return sigmoid_prime(x)
    elif func == relu:
        return relu_prime(x)

def softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
def relu_prime(x):
    return np.where(x > 0, 1, 0)

def sigmoid_prime(x):
    return np.exp(-x)/((1 + np.exp(-x))**2)


def cat_cross_entropy(inputs, truths):
    batch_size = len(inputs)
    clipped_inputs = np.clip(inputs, 1e-7, 1-1e-7)

    if len(truths.shape) == 1:
        true_estimates = clipped_inputs[range(batch_size), truths]
    else:
        true_estimates = np.sum(clipped_inputs * truths, axis=1)
    
    logged_true_estimates = -np.log(true_estimates)

    return np.mean(logged_true_estimates)

def mean_sq_error(inputs, truths): # truths needs to be a two dimensional list of the correct activations for every sample

    error_matrix = np.square(inputs - truths)
    sum_vector = np.sum(error_matrix, axis = 1)

    return np.mean(sum_vector)