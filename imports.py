import numpy as np

class DenseLayer:
    
    def __init__(self, n_inputs, n_neurons, activation):
        self.weights = .1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
    
    def forward(self, inputs):
        matrix_calc = np.dot(inputs, self.weights) + self.biases
        return activation_func(matrix_calc, self.activation)
        


def activation_func(inputs, func_type):
    if func_type == "relu":
        return np.maximum(0, inputs)
    
    elif func_type == 'sigmoid':
        sigmoid = lambda x: 1 / (1 + np.exp(x))
        return sigmoid(inputs)
    
    elif func_type == 'softmax':
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cat_cross_entropy(inputs, truths):
    batch_size = len(inputs)
    clipped_inputs = np.clip(inputs, 1e-7, 1-1e-7)

    if len(truths.shape) == 1:
        true_estimates = clipped_inputs[range(batch_size), truths]
    else:
        true_estimates = np.sum(clipped_inputs * truths, axis=1)
    
    logged_true_estimates = -np.log(true_estimates)

    return np.mean(logged_true_estimates)