import numpy as np

class DenseLayer:
    
    def __init__(self, n_inputs, n_neurons, activation_func):
        self.weights = .1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation_func = activation_func

    
    def forward(self, inputs):
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_func(self.z)
        return self.a
    
    def back(self, inputs, truths, rate):

        # assumes loss is MSE:
        weight_deltas = np.dot(inputs.T, act_prime(self.activation_func, self.z) * 2*(self.a - truths))
        self.weights -= (weight_deltas * rate) 

        bias_deltas = np.mean(act_prime(self.activation_func, self.z) * 2 * (self.a - truths), axis = 0)
        self.biases -= bias_deltas * rate

        


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