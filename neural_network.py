#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from time import time
from matplotlib.gridspec import GridSpec


class XORNeuralNetwork:
    def __init__(self, learning_rate=0.1):
        self.weights = np.random.normal(0, 1, (2, 1))
        self.bias = np.random.normal(0, 1, 1)
        self.learning_rate = learning_rate
        self.mse_history = []
        self.weight_history = []
        self.bias_history = []
        self.accuracy_history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        weighted_sum = np.dot(X, self.weights) + self.bias
        return self.sigmoid(weighted_sum)

    def backward_propagation(self, X, y, output):
        error = y - output
        delta = error * self.sigmoid_derivative(output)

        self.weights += self.learning_rate * np.dot(X.T, delta)
        self.bias += self.learning_rate * np.sum(delta)

        return error

    def train(self, X, y, epochs=10000):
        epoch_times = []

        for epoch in range(epochs):
            epoch_start = time()

            output = self.forward_propagation(X)
            error = self.backward_propagation(X, y, output)

            # Record various metrics
            mse = np.mean(error**2)
            self.mse_history.append(mse)
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias.copy())
            self.accuracy_history.append(self.calculate_accuracy(X, y))

            epoch_end = time()
            epoch_times.append(epoch_end - epoch_start)

        return np.mean(epoch_times)

    def predict(self, X):
        predictions = self.forward_propagation(X)
        return (predictions > 0.5).astype(int)

    def calculate_accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100


def plot_decision_boundary(nn, X, y):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Make predictions for each point in the mesh
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
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
    print(f"Bias: {nn.bias[0]}")
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
    # Prepare XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train with single learning rate and show comprehensive analysis
    nn = XORNeuralNetwork(learning_rate=0.5)
    avg_epoch_time = nn.train(X, y)

    # Print detailed analysis
    print_detailed_analysis(nn, avg_epoch_time)

    # Show comprehensive visualizations
    plot_comprehensive_analysis(nn, X, y)

    # Compare different learning rates
    learning_rates = [0.1, 0.5, 1.0]
    plot_learning_rate_comparison(learning_rates, X, y)


if __name__ == "__main__":
    main()
