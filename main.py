#!/usr/bin/env python

import numpy as np

from neural_network.nn.XOR import XORNeuralNetwork, print_detailed_analysis
from neural_network.nn.enchancedXOR import EnhancedXORNeuralNetwork
from neural_network.plots.metrics import (
    plot_comprehensive_analysis,
    plot_learning_rate_comparison,
)
from neural_network.report.report import generate_report, save_report


def run_enchanced_xor():
    # initial dataset (XOR logic gate)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # train the network and analyse the results
    nn = EnhancedXORNeuralNetwork(learning_rate=0.5)
    avg_epoch_time = nn.train(X, y)

    # print_detailed_analysis(nn, avg_epoch_time)
    plot_comprehensive_analysis(nn, X, y)

    # compare different learning rates
    learning_rates = [0.1, 0.5, 1.0]
    plot_learning_rate_comparison(EnhancedXORNeuralNetwork, learning_rates, X, y)


def run_xor():
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


def report():
    report = generate_report()
    save_report(report)


if __name__ == "__main__":
    report()
    # run_xor()
    # run_enchanced_xor()
