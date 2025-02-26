import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the dataset from CSV file
data = pd.read_csv('Question3_Final_CP.csv')

# Prepare the features (X) and target (y)
X = data[['X1', 'X2', 'X3']].values  # 3 input features
y = data['Y'].values.reshape(-1, 1)  # Binary output (0 or 1)

# Normalize the features (Z-score normalization: Z = (X - mu)/std)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Add intercept term (bias) to X
X_normalized = np.c_[np.ones(X_normalized.shape[0]), X_normalized]  # Shape: (n_samples, 4) with bias

# Initial theta parameters (0 for all, including bias)
n_features = X_normalized.shape[1]  # 4 (1 bias + 3 features)
theta = np.zeros((n_features, 1))  # Initial theta 2D: [[0], [0], [0], [0]]


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Cost function with L2 regularization
def compute_cost(X, y, theta, lambda_reg):
    m = len(y)
    h = sigmoid(X.dot(theta))  # Predictions
    # Cost without regularization
    cost = (-1 / m) * (y.T.dot(np.log(h + 1e-15)) + (1 - y).T.dot(np.log(1 - h + 1e-15)))
    # Add L2 regularization (ridge) - exclude theta[0] (bias) from regularization
    reg_term = (lambda_reg / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost[0][0] + reg_term  # Return scalar


# Gradient descent with L2 regularization
def gradient_descent(X, y, theta, alpha, lambda_reg, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        h = sigmoid(X.dot(theta))  # Predictions
        # Gradient without regularization
        gradient = (1 / m) * X.T.dot(h - y)  # Shape: (n_features, 1)
        # Add L2 regularization gradient (exclude theta[0])
        gradient[1:] += (lambda_reg / m) * theta[1:]
        # Update theta
        theta = theta - alpha * gradient
        cost = compute_cost(X, y, theta, lambda_reg)
        cost_history.append(cost)

    return theta, cost_history


# Parameters for different cases
cases = [
    (100, 0.1, 0.1),  # N=100, alpha=0.1, lambda=0.1
    (1000, 0.2, 1),  # N=1000, alpha=0.2, lambda=1
    (10000, 0.3, 10)  # N=10000, alpha=0.3, lambda=10
]

# Run logistic regression for each case
for n_iterations, alpha, lambda_reg in cases:
    theta_init = np.zeros((n_features, 1))
    final_theta, _ = gradient_descent(X_normalized, y, theta_init, alpha, lambda_reg, n_iterations)

    # Compute cost (rounded up to 2 decimal places after floating point)
    cost = compute_cost(X_normalized, y, final_theta, lambda_reg)
    cost_rounded = round(cost, 2)  # Round to 2 decimal places

    # Find maximum theta value (rounded up to 2 decimal places after floating point)
    max_theta = np.max(np.abs(final_theta))  # Use absolute value for maximum
    max_theta_rounded = round(max_theta, 2)  # Round to 2 decimal places

    print(f"N={n_iterations}, alpha={alpha}, lambda={lambda_reg}")
    print(f"Cost function (rounded up to 2 digits after floating point): {cost_rounded}")
    print(f"Optimal theta parameter maximum value (rounded up to 2 digits after floating point): {max_theta_rounded}\n")

# Special case: After 10,000 iterations, alpha=0.3, lambda=10, predict first 10 rows with threshold=0.5
theta_final, _ = gradient_descent(X_normalized, y, np.zeros((n_features, 1)), 0.3, 10, 10000)
predictions = sigmoid(X_normalized.dot(theta_final))  # Predict probabilities for all rows
first_10_predictions = (predictions[:10] >= 0.5).astype(int)  # Apply threshold 0.5, convert to 0 or 1
number_of_ones = np.sum(first_10_predictions)  # Count number of 1s in first 10 rows

print(f"Number of ones in the first 10 rows of predictions (threshold=0.5): {int(number_of_ones)}")
#N=100, alpha=0.1, lambda=0.1 Cost function (rounded up to 2 digits after floating point): 0.28 Optimal theta parameter maximum value (rounded up to 2 digits after floating point): 1.61
#N=1000, alpha=0.2, lambda=1 Cost function (rounded up to 2 digits after floating point): 0.16 Optimal theta parameter maximum value (rounded up to 2 digits after floating point): 4.59
#N=10000, alpha=0.3, lambda=10 Cost function (rounded up to 2 digits after floating point): 0.33 Optimal theta parameter maximum value (rounded up to 2 digits after floating point): 2.02
#Number of ones in the first 10 rows of predictions (threshold=0.5): 6