#!/usr/bin/env python

import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

from neural_network.nn.XOR import print_detailed_analysis
from neural_network.plots.metrics import (
    plot_comprehensive_analysis,
    plot_learning_rate_comparison,
)


class EnhancedXORNeuralNetwork:
    def __init__(self, hidden_size=4, learning_rate=0.1):
        # network architecture
        self.input_size = 2  # input layer size (2 neurons)
        self.hidden_size = hidden_size  # hidden layer size
        self.output_size = 1  # output layer size (1 neuron)
        self.learning_rate = learning_rate

        # initialize weights
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(
            2.0 / self.input_size
        )
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(
            2.0 / self.hidden_size
        )

        # initialize biases
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))

        # tracking for plotting
        self.mse_history = []
        self.weight_history = {"weights1": [], "weights2": []}
        self.bias_history = {"bias1": [], "bias2": []}
        self.accuracy_history = []

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU function."""
        return (x > 0).astype(float)

    def sigmoid(self, x):
        """Sigmoid activation for output layer."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)

    def forward_propagation(self, X):
        """Forward pass through the network"""

        # apply ReLU activation to hidden layer
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)

        # sigmoid activation for output layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.output = self.sigmoid(self.z2)

        return self.output

    def backward_propagation(self, X, y, output):
        """Backward pass to update weights and biases.

        Args:
            X: Input data.
            y: Target data.
            output: Output of the network.
        Returns:
            Error of the output layer.
        """
        # error of the ouput layer
        delta2 = (output - y) * self.sigmoid_derivative(output)

        # error of the hidden layer
        delta1 = np.dot(delta2, self.weights2.T) * self.relu_derivative(self.a1)

        # output layer adjustments
        self.weights2 -= self.learning_rate * np.dot(self.a1.T, delta2)
        self.bias2 -= self.learning_rate * np.sum(delta2, axis=0, keepdims=True)

        # hidden layer adjustments
        self.weights1 -= self.learning_rate * np.dot(X.T, delta1)
        self.bias1 -= self.learning_rate * np.sum(delta1, axis=0, keepdims=True)

        return y - output

    def train(self, X, y, epochs=5000):
        """Train the network."""
        epoch_times = []

        for epoch in range(epochs):
            epoch_start = perf_counter()

            # backward and forward propagation
            output = self.forward_propagation(X)
            error = self.backward_propagation(X, y, output)

            mse = np.mean(error**2)
            self.mse_history.append(mse)
            self.accuracy_history.append(self.calculate_accuracy(X, y))

            # metrics for visualization
            mse = np.mean(error**2)
            self.mse_history.append(mse)

            self.weight_history["weights1"].append(self.weights1.copy())
            self.weight_history["weights2"].append(self.weights2.copy())

            self.bias_history["bias1"].append(self.bias1)
            self.bias_history["bias2"].append(self.bias2)

            self.accuracy_history.append(self.calculate_accuracy(X, y))
            # ------------------------

            epoch_end = perf_counter()
            epoch_times.append(epoch_end - epoch_start)

            # stop early if converged
            if self.accuracy_history[-1] == 100.0 and mse < 1e-5:
                print(f"Converged at epoch {epoch}")
                break

        return np.mean(epoch_times)

    def predict(self, X):
        """Make predictions using the trained network."""
        predictions = self.forward_propagation(X)
        return (predictions > 0.5).astype(int)

    def calculate_accuracy(self, X, y):
        """Calculate prediction accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100


def compare_networks():
    """Compare original and enhanced networks"""
    # xor logic gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # enchanced xor
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
    # initial dataset (XOR logic gate)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # train the network and analyse the results
    nn = EnhancedXORNeuralNetwork(learning_rate=0.5)
    avg_epoch_time = nn.train(X, y)

    print_detailed_analysis(nn, avg_epoch_time)
    plot_comprehensive_analysis(nn, X, y)

    # compare different learning rates
    learning_rates = [0.1, 0.5, 1.0]
    plot_learning_rate_comparison(EnhancedXORNeuralNetwork, learning_rates, X, y)
