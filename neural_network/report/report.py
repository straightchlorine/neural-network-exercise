#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from time import strftime
from io import BytesIO
import base64
from time import strftime
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from neural_network.nn.XOR import XORNeuralNetwork
from neural_network.nn.enchancedXOR import EnhancedXORNeuralNetwork
from neural_network.plots.metrics import (
    plot_comprehensive_analysis,
    plot_decision_boundary,
)


def generate_plot_base64(fig):
    """Convert a matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_report(learning_rates=[0.1, 0.5, 1.0], epochs=10000):
    """Generate an enhanced report with comprehensive analysis and visualizations"""

    def generate_plot_base64(fig):
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    network_results = {"XOR": {}, "Enhanced XOR": {}}
    visualization_data = {"XOR": {}, "Enhanced XOR": {}}

    for lr in learning_rates:
        xor_nn = XORNeuralNetwork(learning_rate=lr)
        enhanced_nn = EnhancedXORNeuralNetwork(learning_rate=lr)

        xor_time = xor_nn.train(X, y, epochs=epochs)
        enhanced_time = enhanced_nn.train(X, y, epochs=epochs)

        for name, nn, time in [
            ("XOR", xor_nn, xor_time),
            ("Enhanced XOR", enhanced_nn, enhanced_time),
        ]:
            # training histories for visualization
            visualization_data[name][lr] = {
                "mse_history": nn.mse_history,
                "accuracy_history": nn.accuracy_history,
                "weight_history": nn.weight_history,
                "bias_history": nn.bias_history,
            }

            # convergence
            early_convergence = next(
                (i for i, mse in enumerate(nn.mse_history) if mse < 0.1), epochs
            )
            final_convergence = next(
                (i for i, mse in enumerate(nn.mse_history) if mse < 0.01), epochs
            )

            # stability metrics
            accuracy_window = nn.accuracy_history[-1000:]
            mse_window = nn.mse_history[-1000:]

            # learning dynamics
            accuracy_changes = np.diff(nn.accuracy_history)
            positive_improvements = np.sum(accuracy_changes > 0)
            negative_improvements = np.sum(accuracy_changes < 0)

            analysis = generate_plot_base64(
                plot_comprehensive_analysis(nn, X, y, show=False)
            )

            network_results[name][lr] = {
                "analysis": analysis,
                "network": nn,
                "training_time": time,
                "convergence": {
                    "early_convergence": early_convergence,
                    "final_convergence": final_convergence,
                    "convergence_rate": (1.0 - nn.mse_history[-1]) / epochs,
                },
                "final_metrics": {
                    "accuracy": nn.accuracy_history[-1],
                    "mse": nn.mse_history[-1],
                    "stability_score": 1.0
                    / (1.0 + np.std(accuracy_window) + np.std(mse_window)),
                    "accuracy_mse_correlation": np.corrcoef(
                        accuracy_changes, np.diff(nn.mse_history)
                    )[0, 1],
                },
                "training_dynamics": {
                    "positive_improvements": positive_improvements,
                    "negative_improvements": negative_improvements,
                    "improvement_ratio": positive_improvements
                    / (negative_improvements + 1),
                    "average_improvement": np.mean(np.abs(accuracy_changes)),
                },
            }

    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle("Training Metrics Comparison", fontsize=16)

    # mse evolution for both networks
    for name in ["XOR", "Enhanced XOR"]:
        for lr in learning_rates:
            axes[0, 0].plot(
                visualization_data[name][lr]["mse_history"], label=f"{name} (LR={lr})"
            )
    axes[0, 0].set_title("MSE Evolution")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Mean Squared Error")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # accuracy evolution for both networks
    for name in ["XOR", "Enhanced XOR"]:
        for lr in learning_rates:
            axes[0, 1].plot(
                visualization_data[name][lr]["accuracy_history"],
                label=f"{name} (LR={lr})",
            )
    axes[0, 1].set_title("Accuracy Evolution")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # convergnce comparison
    conv_data = [
        [
            network_results[net][lr]["convergence"]["final_convergence"]
            for lr in learning_rates
        ]
        for net in ["XOR", "Enhanced XOR"]
    ]
    axes[1, 0].bar(np.arange(len(learning_rates)) - 0.2, conv_data[0], 0.4, label="XOR")
    axes[1, 0].bar(
        np.arange(len(learning_rates)) + 0.2, conv_data[1], 0.4, label="Enhanced XOR"
    )
    axes[1, 0].set_title("Convergence Time Comparison")
    axes[1, 0].set_xlabel("Learning Rate")
    axes[1, 0].set_xticks(range(len(learning_rates)))
    axes[1, 0].set_xticklabels(learning_rates)
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # accuracy comparison
    acc_data = [
        [network_results[net][lr]["final_metrics"]["accuracy"] for lr in learning_rates]
        for net in ["XOR", "Enhanced XOR"]
    ]
    axes[1, 1].bar(np.arange(len(learning_rates)) - 0.2, acc_data[0], 0.4, label="XOR")
    axes[1, 1].bar(
        np.arange(len(learning_rates)) + 0.2, acc_data[1], 0.4, label="Enhanced XOR"
    )
    axes[1, 1].set_title("Final Accuracy Comparison")
    axes[1, 1].set_xlabel("Learning Rate")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_xticks(range(len(learning_rates)))
    axes[1, 1].set_xticklabels(learning_rates)
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    comparison_plot_base64 = generate_plot_base64(fig1)
    plt.close(fig1)

    # Generate the report with results
    report = f"""# Raport z  list 6 i 7
Wygenerowany: {strftime('%Y-%m-%d %H:%M:%S')}

## 1. Wstęp

Sprawozdanie skupia się na rozwiązaniu problemu z listy 6 oraz 7. Zakładają one utworzenie sieci neuronowej, zdolnej do nauki operatora XOR:

| Input 1 | Input 2 | Expected Output |
|---------|---------|----------------|
| 0       | 0       | 0              |
| 0       | 1       | 1              |
| 1       | 0       | 1              |
| 1       | 1       | 0              |

Celem tego sprawozdania jest przedstawienie wizualizacji sieci neuronowej oraz porównanie wyników dla różnych wartości współczynnika uczenia. Zaprezentowana również zostanie inna sieć neuronowa, wykorzystująca warstwę ukrytą, która pozwala na rozwiązanie problemu XOR, który nie jest liniowo separowalny.

## 2. Trenowanie i wyniki sieci neuronowej

### 2.1 Sieć bez warstwy ukrytej

Sieć neuronowa została przetestowana dla różnych współczynnów uczenia, aby zobrazować jego wpływ na wyniki.

Poniżej widoczne są wyniki dla sieci neuronowej bez warstwy ukrytej:

"""

    for lr in learning_rates:
        report += f"""
#### Współczynnik uczenia równy {lr}
![Analiza sieci neuronowej dla współczynnika uczenia {lr}](data:image/png;base64,{network_results['XOR'][lr]['analysis']})
"""

    report += """
Przeglądając wyniki, zauważyć można, że po lekkich wahaniach, dokładnośc algorytmu pozostaje na poziomie 50%,
możliwe jest jednak osiągnięcie 75%, jeżeli sieci uda się napotkać takowe rozwiązanie.

Jest to spowodowane tym, że problem XOR nie jest liniowo separowalny - nie możliwe jest odnalezienie prostej, która podzieli przestrzeń na dwie klasy, tak, aby każdy z punktów był poprawnie sklasyfikowany.

Najlepszym scenariuszem w przypadku to właśnie 75% dokładności, co oznacza, że sieć neuronowa jest w stanie poprawnie sklasyfikować 3 z 4 punktów. W większości przypadków dla rozpatrywanej liczby iteracji, zatrzymuje się ona na 50%

Warto również zwrócić uwagę na wykres średniego błędu kwadratowego (MSE) oraz wag i biasów. Sieć neuronowa za pośrednictem procesów propagacji, dąży do minimalizacji błędu. Warto jednak zauważyć, że wskaźnik MSE gwałtownie maleje i w pewnym momencie niemalże się zatrzymuje. Nie jest on jednak w stanie osiągnąć wartości 0, co świadczy o tym, że sieć nie jest w stanie dokonać odpowiedniej klasyfikacji.

### 2.1 Sieć z warstwą ukrytą

Podobnie, jak w poprzednim przypadku, sieć neuronowa z warstwą ukrytą została przetestowana dla różnych wartości współczynnika uczenia: {learning_rates}:
    """

    for lr in learning_rates:
        report += f"""
##### Współczynnik uczenia równy {lr}
![Analize sieci neuronowej z warstwą ukrytą dla współczynnika uczenia {lr}](data:image/png;base64,{network_results['Enhanced XOR'][lr]['analysis']})"""

    report += f"""
## 3. Porównanie zbieżności

Ze względu na naturę problemu, sieć z warstwą ukrytą (EXOR) osiąga znacznie lepsze wyniki niż sieć bez warstwy ukrytej (XOR). Dzięki zastosowaniu warstwy ukrytej, sieć neuronowa jest w stanie nauczyć się nieliniowych zależności, które występują w przypadku problemu XOR:

- Stosunek szybkości zbieżności (EXOR)/(XOR): {np.mean([
    network_results['Enhanced XOR'][lr]['convergence']['convergence_rate'] /
    network_results['XOR'][lr]['convergence']['convergence_rate']
    for lr in learning_rates]):.2f}x

## 4. Podsumowanie wyników
- Najwyższa dokładność dla XOR: {max([
    network_results['XOR'][lr]['final_metrics']['accuracy']
    for lr in learning_rates]):.2f}%
- Najwyższa dokładność dla EXOR: {max([
    network_results['Enhanced XOR'][lr]['final_metrics']['accuracy']
    for lr in learning_rates]):.2f}%

## 5. Stabilność trenowania
- Stosunek stabilności (EXOR)/(XOR) {np.mean([
    network_results['Enhanced XOR'][lr]['final_metrics']['stability_score'] /
    network_results['XOR'][lr]['final_metrics']['stability_score']
    for lr in learning_rates]):.2f}x lepsza stabilność

## 6. Analiza zbieżności

| Network Type | Learning Rate | Early Convergence | Final Convergence | Convergence Rate |
|-------------|---------------|-------------------|-------------------|------------------|
"""

    for name in ["XOR", "Enhanced XOR"]:
        for lr in learning_rates:
            metrics = network_results[name][lr]["convergence"]
            report += f"| {name} | {lr} | {metrics['early_convergence']} | "
            report += f"{metrics['final_convergence']} | "
            report += f"{metrics['convergence_rate']:.6f} |\n"

    report += """
## 7. Końcowe wskaźniki sieci neuronowej

| Network Type | Learning Rate | Final Accuracy | Final MSE | Stability Score | MSE-Accuracy Correlation |
|-------------|---------------|----------------|-----------|-----------------|--------------------------|
"""

    for name in ["XOR", "Enhanced XOR"]:
        for lr in learning_rates:
            metrics = network_results[name][lr]["final_metrics"]
            report += f"| {name} | {lr} | {metrics['accuracy']:.2f}% | "
            report += f"{metrics['mse']:.6f} | {metrics['stability_score']:.4f} | {metrics['accuracy_mse_correlation']:.4f}\n"

    report += """
## 8. Analiza współczynnika nauczania

### 8.1. Optymalne wartości
     """
    for name in ["XOR", "Enhanced XOR"]:
        best_lr = max(
            learning_rates,
            key=lambda lr: network_results[name][lr]["final_metrics"]["accuracy"],
        )
        report += f"\n- {name} Network: {best_lr} "
        report += f"(Accuracy: {network_results[name][best_lr]['final_metrics']['accuracy']:.2f}%) "
        report += (
            f"(MSE: {network_results[name][best_lr]['final_metrics']['mse']:.6f})\n\n"
        )

    report += f"""
### 8.2. Porównanie zbieżności
![Porównanie zbieżności](data:image/png;base64,{comparison_plot_base64})
"""
    return report, network_results


def save_report(report, filename="sprawozdanie.md"):
    """Save the enhanced report to a markdown file"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report[0])
    print(f"Enhanced report saved to {filename}")
