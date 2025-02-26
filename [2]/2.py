
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Read the dataset from CSV file (use the filename as is, since it's in the same directory)
data = pd.read_csv('Question2_Dataset.csv')

# Prepare the features (X) and target (y)
X = data[['X1', 'X2', 'X1^2', 'X1^3', 'X2^2', 'X2^3', 'X1*X2', 'X1^2*X2']].values
y = data['Y'].values.reshape(-1, 1)

# Normalize the features (Z-score normalization)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_normalized, y)

# Get the initial theta parameters (including bias)
initial_theta = np.zeros(X.shape[1] + 1)  # +1 for bias term (9 parameters total: 1 bias + 8 features)
theta = np.concatenate([model.intercept_, model.coef_.flatten()])

# Learning rate
learning_rate = 0.1


# Function to calculate cost (Mean Squared Error)
def calculate_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta[1:]) + theta[0]  # Shape: (m,)
    cost = (1 / (2 * m)) * np.sum((predictions - y.flatten()) ** 2)
    return cost


# Gradient descent implementation
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)  # Number of samples
    n = X.shape[1]  # Number of features
    cost_history = []

    # Ensure y is flattened to avoid shape mismatch
    y = y.flatten()  # Now y has shape (m,)

    for _ in range(iterations):
        # Calculate predictions (shape: (m,))
        predictions = X.dot(theta[1:]) + theta[0]  # theta[1:] has shape (n,), X has shape (m,n)
        errors = predictions - y  # Shape: (m,)

        # Update bias (theta[0]) - sum over all examples
        theta[0] = theta[0] - (learning_rate / m) * np.sum(errors)

        # Update feature coefficients (theta[1:]) - calculate gradient correctly
        gradient_features = (1 / m) * X.T.dot(errors)  # Shape: (n,) where n=8
        theta[1:] = theta[1:] - learning_rate * gradient_features  # Both shapes (n,)

        cost = calculate_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history


# Run gradient descent for different iterations
n_10_results = gradient_descent(X_normalized, y, initial_theta.copy(), learning_rate, 10)
n_100_results = gradient_descent(X_normalized, y, initial_theta.copy(), learning_rate, 100)
n_1000_results = gradient_descent(X_normalized, y, initial_theta.copy(), learning_rate, 1000)

# Print results in the required format (rounded to integers)
print("n=10")
print("Cost Function (Round):", round(calculate_cost(X_normalized, y, n_10_results[0])))
print("Optimal Theta parameter (Round):", [round(x) for x in n_10_results[0]])

print("\nn=100")
print("Cost Function (Round):", round(calculate_cost(X_normalized, y, n_100_results[0])))
print("Optimal Theta parameter (Round):", [round(x) for x in n_100_results[0]])

print("\nn=1000")
print("Cost Function (Round):", round(calculate_cost(X_normalized, y, n_1000_results[0])))
print("Optimal Theta parameter (Round):", [round(x) for x in n_1000_results[0]])


#n=10
#Cost Function (Round): 895241
#Optimal Theta parameter (Round): [2167, -54, 839, -47, -31, 1001, 1080, 436, 283]

#n=100
#Cost Function (Round): 42271
#Optimal Theta parameter (Round): [3328, 50, 230, 125, 192, 1244, 1897, 18, 63]

#n=1000
#Cost Function (Round): 1261
#Optimal Theta parameter (Round): [3328, -17, -520, 119, 228, 1231, 2651, -64, 200]
