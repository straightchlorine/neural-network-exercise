#!/usr/bin/env python

from time import perf_counter

import numpy as np

from neural_network.plots.metrics import (
    plot_comprehensive_analysis,
    plot_learning_rate_comparison,
)


class XORNeuralNetwork:
    def __init__(self, learning_rate=0.1):
        self.weights = np.random.normal(0, 1, (2, 1))  # weights dim(2, 1) - 2 neurons
        self.bias = np.random.normal(0, 1)  # bias dim(1), a scalar - 1 neuron
        self.learning_rate = learning_rate
        self.mse_history = []
        self.weight_history = []
        self.bias_history = []
        self.accuracy_history = []

    def sigmoid(self, x):
        """Activation function (sigmoid function)."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, sigmoid_output):
        """Derivative of the sigmoid function.

        Args:
            sigmoid_output: Output of the sigmoid function.
        """
        return sigmoid_output * (1 - sigmoid_output)

    def forward_propagation(self, X):
        """Calculate the weighted sum of the input and call sigmoid function
        in order to get the output of the neuron.

        Args:
            X: Input data.
        """
        weighted_sum = np.dot(X, self.weights) + self.bias
        return self.sigmoid(weighted_sum)

    def backward_propagation(self, X, y, output):
        """Calculate the error and update weights and biases.

        Args:
            X: Input data.
            y: Target data.
            output: Output of the neuron.
        """
        error = y - output
        delta = error * self.sigmoid_derivative(output)

        self.weights += self.learning_rate * np.dot(X.T, delta)  # gradient of weights
        self.bias += self.learning_rate * np.sum(delta)  # gradient of bias

        return error

    def train(self, X, y, epochs=10000):
        """Train the network.

        Training is an iterative process of updating weights and biases of the
        network in order to minimize the error.

        In other words it is a repeated forward and backward propagation, first
        output is calculated based on input data, then error is calculated
        against the target data and weights and biases are updated based on the
        gradient of the error, input data and the learning rate.

        Args:
            X: Input data.
            y: Target data.
            epochs: Number of iterations to train the network.

        Returns:
            Average time per epoch.
        """
        epoch_times = []

        for _ in range(epochs):
            epoch_start = perf_counter()

            output = self.forward_propagation(X)
            error = self.backward_propagation(X, y, output)

            # update metrics for visualizations
            mse = np.mean(error**2)  # mean squared error
            self.mse_history.append(mse)
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)
            self.accuracy_history.append(self.calculate_accuracy(X, y))
            # --------------------------------

            epoch_end = perf_counter()
            epoch_times.append(epoch_end - epoch_start)

        return np.mean(epoch_times)

    def predict(self, X):
        """Sigmoid output is converted here to binary prediction.

        Based on the received output of the neuron, if it is greater than 0.5
        network predicts it as 1, otherwise as 0.

        Args:
            X: Input data.

        Returns:
            Binary predictions.
        """
        predictions = self.forward_propagation(X)
        return (predictions > 0.5).astype(int)

    def calculate_accuracy(self, X, y):
        """Calculate the accuracy of the model.

        Args:
            X: Input data.
            y: Target data.

        Returns:
            Accuracy of the model in %.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100


def print_detailed_analysis(nn, avg_epoch_time):
    print("\nDetailed Analysis:")
    print("-" * 50)
    print("Final Model Parameters:")
    print(f"Weights: {nn.weights.flatten()}")
    print(f"Bias: {nn.bias}")
    print("\nTraining Metrics:")
    print(f"Initial MSE: {nn.mse_history[0]:.6f}")
    print(f"Final MSE: {nn.mse_history[-1]:.6f}")
    print(f"MSE Improvement: {(1 - nn.mse_history[-1]/nn.mse_history[0])*100:.2f}%")
    print(f"Final Accuracy: {nn.accuracy_history[-1]:.2f}%")
    print(f"Average time per epoch: {avg_epoch_time*1000:.3f} ms")

    # convergence metrics
    convergence_threshold = 0.01
    converged_epoch = next(
        (i for i, mse in enumerate(nn.mse_history) if mse < convergence_threshold),
        len(nn.mse_history),
    )
    print("\nConvergence Analysis:")
    print(f"Epochs to reach MSE < {convergence_threshold}: {converged_epoch}")


def main():
    # initial dataset (XOR logic gate)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # train the network and analyse the results
    nn = XORNeuralNetwork(learning_rate=0.5)
    avg_epoch_time = nn.train(X, y)

    print_detailed_analysis(nn, avg_epoch_time)
    plot_comprehensive_analysis(nn, X, y)

    # compare different learning rates
    learning_rates = [0.1, 0.5, 1.0]
    plot_learning_rate_comparison(XORNeuralNetwork, learning_rates, X, y)


if __name__ == "__main__":
    main()
