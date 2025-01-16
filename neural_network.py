#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from matplotlib.gridspec import GridSpec


class XORNeuralNetwork:
    def __init__(self, learning_rate=0.1):
        self.weights = np.random.normal(0, 1, (2, 1))  # weights dim(2, 1)
        self.bias = np.random.normal(0, 1)  # bias dim(1), a scalar
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

    #  ----- List 7 -----

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


def plot_decision_boundary(nn, X, y):
    """Plot the decision boundary of the neural network.

    Args:
        nn: Neural network model.
        X: Input data.
        y: Target data.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    # predictions for each point in mesh
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")


def plot_comprehensive_analysis(nn, X, y):
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Plot 1: MSE over epochs
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(nn.mse_history)
    ax1.set_title("Mean Squared Error over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.grid(True)

    # Plot 2: Accuracy over epochs
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(nn.accuracy_history)
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)

    # Plot 3: Weight evolution
    ax3 = fig.add_subplot(gs[1, 0])
    weight_history = np.array(nn.weight_history)
    ax3.plot(weight_history[:, 0], label="Weight 1")
    ax3.plot(weight_history[:, 1], label="Weight 2")
    ax3.set_title("Weight Evolution")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Weight Value")
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Bias evolution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(nn.bias_history)
    ax4.set_title("Bias Evolution")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Bias Value")
    ax4.grid(True)

    # Plot 5: Decision boundary
    ax5 = fig.add_subplot(gs[2, :])
    plot_decision_boundary(nn, X, y)
    ax5.set_title("Decision Boundary")

    plt.tight_layout()
    plt.show()


def plot_learning_rate_comparison(learning_rates, X, y, epochs=10000):
    mse_histories = []
    acc_histories = []

    for lr in learning_rates:
        nn = XORNeuralNetwork(learning_rate=lr)
        nn.train(X, y, epochs=epochs)
        mse_histories.append(nn.mse_history)
        acc_histories.append(nn.accuracy_history)

    # Plot MSE comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i, lr in enumerate(learning_rates):
        plt.plot(mse_histories[i], label=f"LR = {lr}")
    plt.title("MSE Comparison for Different Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy comparison
    plt.subplot(1, 2, 2)
    for i, lr in enumerate(learning_rates):
        plt.plot(acc_histories[i], label=f"LR = {lr}")
    plt.title("Accuracy Comparison for Different Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def print_detailed_analysis(nn, avg_epoch_time):
    print("\nDetailed Analysis:")
    print("-" * 50)
    print(f"Final Model Parameters:")
    print(f"Weights: {nn.weights.flatten()}")
    print(f"Bias: {nn.bias}")
    print(f"\nTraining Metrics:")
    print(f"Initial MSE: {nn.mse_history[0]:.6f}")
    print(f"Final MSE: {nn.mse_history[-1]:.6f}")
    print(f"MSE Improvement: {(1 - nn.mse_history[-1]/nn.mse_history[0])*100:.2f}%")
    print(f"Final Accuracy: {nn.accuracy_history[-1]:.2f}%")
    print(f"Average time per epoch: {avg_epoch_time*1000:.3f} ms")

    # Calculate convergence metrics
    convergence_threshold = 0.01
    converged_epoch = next(
        (i for i, mse in enumerate(nn.mse_history) if mse < convergence_threshold),
        len(nn.mse_history),
    )
    print(f"\nConvergence Analysis:")
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
    plot_learning_rate_comparison(learning_rates, X, y)


if __name__ == "__main__":
    main()
