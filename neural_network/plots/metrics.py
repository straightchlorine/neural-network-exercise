import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


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

    return plt


def plot_weight_bias_by_layer(fig, gs, nn):
    ax_weights = fig.add_subplot(gs[1, 0])

    # input->hidden weights (weights1)
    w1_history = np.array(nn.weight_history["weights1"])
    for i in range(nn.input_size):
        for j in range(nn.hidden_size):
            ax_weights.plot(
                w1_history[:, i, j],
                label=f"W1: Input {i+1}→Hidden {j+1}",
                linestyle="-",
            )

    # hidden->output weights (weights2)
    w2_history = np.array(nn.weight_history["weights2"])
    for i in range(nn.hidden_size):
        ax_weights.plot(
            w2_history[:, i, 0], label=f"W2: Hidden {i+1}→Output", linestyle="--"
        )

    ax_weights.set_title("Weight Evolution")
    ax_weights.set_xlabel("Epoch")
    ax_weights.set_ylabel("Weight Value")
    ax_weights.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_weights.grid(True)

    # ----------------------------------------------------
    ax_bias = fig.add_subplot(gs[1, 1])

    # hidden layer biases
    b1_history = np.array(nn.bias_history["bias1"])
    for i in range(nn.hidden_size):
        ax_bias.plot(b1_history[:, 0, i], label=f"B1: Hidden {i+1}", linestyle="-")

    # output layer bias
    b2_history = np.array(nn.bias_history["bias2"])
    ax_bias.plot(b2_history[:, 0, 0], label="B2: Output", linestyle="--")

    ax_bias.set_title("Bias Evolution")
    ax_bias.set_xlabel("Epoch")
    ax_bias.set_ylabel("Bias Value")
    ax_bias.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_bias.grid(True)
    plt.tight_layout()


def plot_weight_bias_single_layer(fig, gs, nn):
    ax = fig.add_subplot(gs[1, 0])
    weight_history = np.array(nn.weight_history)
    ax.plot(weight_history[:, 0], label="Weight 1")
    ax.plot(weight_history[:, 1], label="Weight 2")
    ax.set_title("Weight Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Value")
    ax.legend()
    ax.grid(True)

    # bias plot
    ax2 = fig.add_subplot(gs[1, 1])
    bias_history = np.array(nn.bias_history)
    ax2.plot(bias_history)
    ax2.set_title("Bias Evolution")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Bias Value")
    ax2.grid(True)


def plot_weight_bias(fig, gs, nn):
    if isinstance(nn.weight_history, dict):
        plot_weight_bias_by_layer(fig, gs, nn)
    else:
        plot_weight_bias_single_layer(fig, gs, nn)


def plot_mse_accuracy(fig, gs, nn):
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(nn.mse_history)
    ax1.set_title("Mean Squared Error over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(nn.accuracy_history)
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)


def plot_comprehensive_analysis(nn, X, y, show=True):
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    plot_weight_bias(fig, gs, nn)
    plot_mse_accuracy(fig, gs, nn)

    # decision boundary
    ax5 = fig.add_subplot(gs[2, :])
    plot_decision_boundary(nn, X, y)
    ax5.set_title("Decision Boundary")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_learning_rate_comparison(nn_class, learning_rates, X, y, epochs=10000):
    mse_histories = []
    acc_histories = []

    for lr in learning_rates:
        nn = nn_class(learning_rate=lr)
        nn.train(X, y, epochs=epochs)
        mse_histories.append(nn.mse_history)
        acc_histories.append(nn.accuracy_history)

    # plot mse
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i, lr in enumerate(learning_rates):
        plt.plot(mse_histories[i], label=f"LR = {lr}")
    plt.title("MSE Comparison for Different Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)

    # plot accuracy
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
