#!/usr/bin/env python

import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt


class EnhancedXORNeuralNetwork:
    def __init__(self, hidden_size=4, learning_rate=0.1):
        # network architecture
        self.input_size = 2  # input layer size (2 nodes)
        self.hidden_size = hidden_size  # hidden layer size
        self.output_size = 1  # output layer size (1 node)
        self.learning_rate = learning_rate

        # Initialize weights with He initialization for better convergence
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(
            2.0 / self.input_size
        )
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(
            2.0 / self.hidden_size
        )

        # Initialize biases with small random values
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))

        # History tracking for visualization
        self.mse_history = []
        self.accuracy_history = []

    def relu(self, x):
        """ReLU activation function - better for deep networks than sigmoid"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)

    def sigmoid(self, x):
        """Sigmoid activation for output layer - good for binary classification"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)

    def forward_propagation(self, X):
        """Forward pass through the network"""
        # First layer with ReLU activation
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)

        # Output layer with sigmoid activation
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.output = self.sigmoid(self.z2)

        return self.output

    def backward_propagation(self, X, y, output):
        """Backward pass to update weights and biases"""
        m = X.shape[0]  # batch size

        # Output layer error
        delta2 = (output - y) * self.sigmoid_derivative(output)

        # Hidden layer error
        delta1 = np.dot(delta2, self.weights2.T) * self.relu_derivative(self.a1)

        # Update weights and biases with momentum
        self.weights2 -= self.learning_rate * np.dot(self.a1.T, delta2)
        self.bias2 -= self.learning_rate * np.sum(delta2, axis=0, keepdims=True)
        self.weights1 -= self.learning_rate * np.dot(X.T, delta1)
        self.bias1 -= self.learning_rate * np.sum(delta1, axis=0, keepdims=True)

        return y - output

    def train(self, X, y, epochs=5000):
        """Train the network"""
        epoch_times = []

        for epoch in range(epochs):
            epoch_start = perf_counter()

            # Forward and backward passes
            output = self.forward_propagation(X)
            error = self.backward_propagation(X, y, output)

            # Calculate and store metrics
            mse = np.mean(error**2)
            self.mse_history.append(mse)
            self.accuracy_history.append(self.calculate_accuracy(X, y))

            epoch_end = perf_counter()
            epoch_times.append(epoch_end - epoch_start)

            # Early stopping if we reach perfect accuracy
            if self.accuracy_history[-1] == 100.0 and mse < 1e-5:
                print(f"Converged at epoch {epoch}")
                break

        return np.mean(epoch_times)

    def predict(self, X):
        """Make predictions using the trained network"""
        predictions = self.forward_propagation(X)
        return (predictions > 0.5).astype(int)

    def calculate_accuracy(self, X, y):
        """Calculate prediction accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100


def compare_networks():
    """Compare original and enhanced networks"""
    # Prepare XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train enhanced network
    enhanced_nn = EnhancedXORNeuralNetwork(hidden_size=4, learning_rate=0.1)
    enhanced_time = enhanced_nn.train(X, y)

    # Print results
    print("\nEnhanced Network Results:")
    print("-" * 50)
    print(f"Final Accuracy: {enhanced_nn.accuracy_history[-1]:.2f}%")
    print(f"Final MSE: {enhanced_nn.mse_history[-1]:.6f}")
    print("\nPredictions:")
    predictions = enhanced_nn.predict(X)
    for inputs, pred, target in zip(X, predictions, y):
        print(f"Input: {inputs}, Predicted: {pred[0]}, Target: {target[0]}")

    # Plot training progress
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(enhanced_nn.mse_history)
    plt.title("MSE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(enhanced_nn.accuracy_history)
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_networks()
