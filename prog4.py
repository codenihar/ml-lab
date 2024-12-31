import numpy as np

# Sigmoid Activation Function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training Data (XOR problem as an example)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
y = np.array([[0], [1], [1], [0]])  # Output (XOR)

# Hyperparameters
epochs = 10000
learning_rate = 0.1
input_layer_neurons = X.shape[1]  # Number of input features
hidden_layer_neurons = 4  # Number of neurons in hidden layer
output_layer_neurons = 1  # Single output neuron

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.rand(input_layer_neurons, hidden_layer_neurons)
weights_hidden_output = np.random.rand(hidden_layer_neurons, output_layer_neurons)
bias_hidden = np.random.rand(1, hidden_layer_neurons)
bias_output = np.random.rand(1, output_layer_neurons)

# Training the Neural Network
for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    
    # Backpropagation
    output_error = y - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)
    
    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_layer_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

# Test the model (after training)
print("Final output after training:")
print(predicted_output)

# Show updated parameters
print("\nUpdated weights and biases after training:")
print("Weights from input to hidden layer:")
print(weights_input_hidden)
print("Weights from hidden to output layer:")
print(weights_hidden_output)
print("Biases for hidden layer:")
print(bias_hidden)
print("Biases for output layer:")
print(bias_output)
