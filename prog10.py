import numpy as np
import matplotlib.pyplot as plt

# Locally Weighted Regression (Simplified)
def locally_weighted_regression(x, X, y, tau):
    """Simplified Locally Weighted Regression"""
    weights = np.exp(-((X - x)**2) / (2 * tau**2))  # Gaussian weights
    weighted_sum = np.sum(weights * y)             # Weighted sum of y
    weight_total = np.sum(weights)                 # Sum of weights
    return weighted_sum / weight_total             # Weighted average

# Generate synthetic data
np.random.seed(42)
X = np.linspace(1, 10, 100)                        # Feature
y = 2 * X + np.random.normal(0, 2, X.shape)        # Target with noise

# Set tau (bandwidth parameter)
tau = 1.0

# Predict using LWR
predictions = [locally_weighted_regression(x, X, y, tau) for x in X]

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label="Data Points", color="blue", alpha=0.6)
plt.plot(X, predictions, label="LWR Curve (tau=1.0)", color="red", linewidth=2)
plt.title("Locally Weighted Regression (Simplified)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
