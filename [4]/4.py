import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Example input (flattened image vectors) - replace with your actual normalized data
X = np.array([[0.25, 0.5, 0.75, 1.0],  # Dog image example (simplified)
              [0.1, 0.3, 0.5, 0.7]])  # Cat image example (simplified)

# Example output (1 for dog, 0 for cat)
y = np.array([[1], [0]])

# Network architecture
input_layer_size = 4  # Number of features
hidden_layer1_size = 7  # First hidden layer neurons
hidden_layer2_size = 5  # Second hidden layer neurons
hidden_layer3_size = 3  # Third hidden layer neurons
output_layer_size = 1  # Binary classification (dog/cat)

# Fixed initial weights and biases (as provided)
W1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
               [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
               [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
               [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]])
b1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])

W2 = np.array([[0.2, 0.3, 0.4, 0.5, 0.6],
               [0.7, 0.8, 0.9, 1.0, 1.1],
               [1.2, 1.3, 1.4, 1.5, 1.6],
               [1.7, 1.8, 1.9, 2.0, 2.1],
               [2.2, 2.3, 2.4, 2.5, 2.6],
               [2.7, 2.8, 2.9, 3.0, 3.1],
               [3.2, 3.3, 3.4, 3.5, 3.6]])
b2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

W3 = np.array([[0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7],
               [0.8, 0.9, 1.0],
               [1.1, 1.2, 1.3],
               [1.4, 1.5, 1.6]])
b3 = np.array([[0.1, 0.2, 0.3]])

W4 = np.array([[0.2], [0.3], [0.4]])
b4 = np.array([[0.1]])

# Training parameters
learning_rate = 0.1
epochs = 10000

# Lists to store metrics for final analysis
a4_history = []
W4_history = []
W3_history = []
loss_history = []

# Training loop
for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)  # Tanh activation for hidden layer 1

    z2 = np.dot(a1, W2) + b2
    a2 = tanh(z2)  # Tanh activation for hidden layer 2

    z3 = np.dot(a2, W3) + b3
    a3 = tanh(z3)  # Tanh activation for hidden layer 3

    z4 = np.dot(a3, W4) + b4
    a4 = sigmoid(z4)  # Sigmoid activation for output layer

    # Compute loss (Mean Absolute Error - MAE)
    error = y - a4
    loss = np.mean(np.abs(error))
    loss_history.append(loss)

    # Store history for final analysis
    a4_history.append(a4.copy())
    W4_history.append(W4.copy())
    W3_history.append(W3.copy())

    # Backpropagation
    d_a4 = error * sigmoid_derivative(a4)  # Derivative of sigmoid for output layer
    d_W4 = np.dot(a3.T, d_a4) * learning_rate
    d_b4 = np.sum(d_a4, axis=0, keepdims=True) * learning_rate

    d_a3 = np.dot(d_a4, W4.T) * tanh_derivative(a3)  # Derivative of Tanh for hidden layer 3
    d_W3 = np.dot(a2.T, d_a3) * learning_rate
    d_b3 = np.sum(d_a3, axis=0, keepdims=True) * learning_rate

    d_a2 = np.dot(d_a3, W3.T) * tanh_derivative(a2)  # Derivative of Tanh for hidden layer 2
    d_W2 = np.dot(a1.T, d_a2) * learning_rate
    d_b2 = np.sum(d_a2, axis=0, keepdims=True) * learning_rate

    d_a1 = np.dot(d_a2, W2.T) * tanh_derivative(a1)  # Derivative of Tanh for hidden layer 1
    d_W1 = np.dot(X.T, d_a1) * learning_rate
    d_b1 = np.sum(d_a1, axis=0, keepdims=True) * learning_rate

    # Update weights and biases
    W4 += d_W4
    b4 += d_b4
    W3 += d_W3
    b3 += d_b3
    W2 += d_W2
    b2 += d_b2
    W1 += d_W1
    b1 += d_b1

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Final predictions (probabilities for each input)
y_pred = a4
print("Final Predictions:", y_pred)

# Compute required metrics after 10,000 epochs
# a4 = [value1, value2, ...] (final activations of output layer, rounded to 3 decimal places)
a4_final = a4_history[-1]  # Last activation values
a4_rounded = np.round(a4_final, 3)  # Round to 3 digits after floating point
print(f"a4 = {a4_rounded.flatten().tolist()}")

# a3_min (minimum activation in hidden layer 3, rounded to 3 decimal places)
a3_final = a3  # Final activations of hidden layer 3
a3_min = np.min(a3_final)
a3_min_rounded = round(a3_min, 3)  # Round to 3 digits after floating point
print(f"a3_min = {a3_min_rounded}")

# W4_max (maximum weight in W4, rounded to 2 decimal places)
W4_max = np.max(np.abs(W4))  # Use absolute value for maximum
W4_max_rounded = round(W4_max, 2)  # Round to 2 digits after floating point
print(f"W4_max = {W4_max_rounded}")

# W3_min (minimum weight in W3, rounded to 2 decimal places)
W3_min = np.min(np.abs(W3))  # Use absolute value for minimum
W3_min_rounded = round(W3_min, 2)  # Round to 2 digits after floating point
print(f"W3_min = {W3_min_rounded}")

# Loss after 10,000 epochs (rounded to 2 decimal places)
loss_final = loss_history[-1]  # Last loss value
loss_rounded = round(loss_final, 2)  # Round to 2 digits after floating point
print(f"Loss after 10000 epochs: {loss_rounded}")

# General Conclusion: Predict class for the inputs (dog or cat)
threshold = 0.5
predictions_binary = (y_pred >= threshold).astype(int)
dog_pred = np.any(predictions_binary == 1)  # If any prediction is 1, predict dog
cat_pred = np.any(predictions_binary == 0)  # If any prediction is 0, predict cat

if dog_pred and not cat_pred:
    conclusion = "NN predicts image of dog"
elif cat_pred and not dog_pred:
    conclusion = "NN predicts image of cat"
else:
    conclusion = "NN can't define correct image class"

print(f"General Conclusion after 10000 epochs: {conclusion}")

'''
Final Predictions: [[0.5] [0.5]] 
a4 = [0.5, 0.5] 
a3_min = 0.999 
W4_max = 0.15 
W3_min = 0.19 
Loss after 10000 epochs: 0.5 
General Conclusion after 10000 epochs: NN predicts image of dog
'''
