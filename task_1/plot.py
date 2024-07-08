import matplotlib.pyplot as plt
import numpy as np
from dataset import classification_error, ellipse_equation
from matplotlib.lines import Line2D


def classification_results(costs, gradient_norms):
    _, axs = plt.subplots(1, 2, figsize=(20, 10))

    axs[0].semilogy(costs, color="blue")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Cost")
    axs[0].set_title("Evolution of the Cost Function")
    axs[0].grid(True)

    axs[1].semilogy(gradient_norms, color="red")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Norm of the Gradient")
    axs[1].set_title("Evolution of the Norm of the Gradient")
    axs[1].grid(True)

    plt.show()


def dataset(title, dataset, *classifiers, **kwargs):
    plt.figure(figsize=(10, 10))
    plt.scatter(dataset[dataset[:, 2] == 1, 0], dataset[dataset[:, 2] == 1, 1], color="blue")
    plt.scatter(dataset[dataset[:, 2] == -1, 0], dataset[dataset[:, 2] == -1, 1], color="red")

    if "misclassified" in kwargs:
        misclassified = kwargs["misclassified"]
        for x in misclassified:
            plt.scatter(x[0], x[1], color="magenta", marker="x", s=100, linewidth=4, label="Misclassified")

    x = np.linspace(np.min(dataset[:, 0]), np.max(dataset[:, 0]), 10000)

    legend_items = []
    for classifier in classifiers:
        a, b, c, d, e = classifier["params"]
        w = np.array([a, b, c, d])
        bias = -(e**2)

        y_pos = np.zeros(len(x))
        y_neg = np.zeros(len(x))

        for i in range(len(x)):
            y_pos[i], y_neg[i] = ellipse_equation(w, bias, x[i])

        plt.plot(x, y_pos, color=classifier["color"], linestyle="--")
        plt.plot(x, y_neg, color=classifier["color"], linestyle="--")
        legend_items.append(Line2D([0], [0], color=classifier["color"], linestyle="--", label=classifier["label"]))

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend(handles=legend_items)
    plt.grid(True)
    plt.show()


def _get_classifier(theta, type):
    w, bias = theta[:4], theta[4]
    a, b, c, d = w
    e = np.sqrt(-bias)
    params = np.array((a, b, c, d, e))

    if type == "real":
        return {
            "params": params,
            "color": "green",
            "label": f"Real Separating Function (${a}x+{b}y+{c}x^2+{d}y^2={e}^2$)",
        }
    else:
        return {
            "params": params,
            "color": "orange",
            "label": f"Estimated Separating Function (${{{a:.2f}}}x+{{{b:.2f}}}y+{{{c:.2f}}}x^2+{{{d:.2f}}}y^2={{{e:.2f}}}$)",
        }


def results(data, theta_hat, real_theta, costs, gradient_magnitude, title, no_plots=False):
    estimated_classifier = _get_classifier(theta_hat, "estimated")
    a, b, c, d, e = estimated_classifier["params"]
    print(f"Estimated parameters: a = {a:.2f}, b = {b:.2f}, c = {c:.2f}, d = {d:.2f}, e = {e:.2f}")

    error, misclassified = classification_error(data, theta_hat)
    print(f"Classification error: {error} %")

    if not no_plots:
        classification_results(costs, gradient_magnitude)

        real_classifier = _get_classifier(real_theta, "real")
        dataset(
            title,
            data,
            real_classifier,
            estimated_classifier,
            misclassified=misclassified,
        )
