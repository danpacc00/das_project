import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def phi(x):
    x = np.array(x)
    x_new = np.hstack((x, x**2))
    return x_new


def separating_function(w, bias, x):
    return np.dot(w, phi(x)) + bias


def ellipse_equation(w, bias, x):
    a, b, c, d = w
    """Ellipse equation: dy^2 +by +cx^2 +ax = e^2

    solution: y1/2 = (-b ± sqrt(b^2 - 4(dx(cx+a))) - e^2) / 2d"""

    y_positive = (-b + np.sqrt(b**2 - 4 * d * (c * x**2 + a * x + bias))) / (2 * d)
    y_negative = (-b - np.sqrt(b**2 - 4 * d * (c * x**2 + a * x + bias))) / (2 * d)

    return y_positive, y_negative


def create_labeled_dataset(w, M, show_plot=False):
    a = w[0]  # Stretch wrt x-axis
    b = w[1]  # Tilt wrt x-axis
    c = w[2]  # Coefficient for x^2. Controls the width of the ellipse
    d = w[3]  # Coefficient for y^2. Controls the height of the ellipse
    e = w[4]

    theta = np.array([a, b, c, d])  # Weights
    bias = -(e**2)  # Bias

    # Plot the separating function as a line
    x_lim = 10
    x_points = np.linspace(-x_lim, x_lim, 10000)

    y_line_pos = np.zeros(len(x_points))
    y_line_neg = np.zeros(len(x_points))

    for i in range(len(x_points)):
        y_line_pos[i] = ellipse_equation(theta, bias, x_points[i])[0]
        y_line_neg[i] = ellipse_equation(theta, bias, x_points[i])[1]

    y_lim = d

    offset = 1.5

    D_1 = np.random.uniform(-x_lim - offset, x_lim + offset, size=(M, 1))
    D_2 = np.random.uniform(-y_lim - offset, y_lim + offset, size=(M, 1))

    D = np.concatenate((D_1, D_2), axis=1)

    # Label the data with the separating function
    labeled_dataset = np.zeros((M, 3))
    plt.figsize = (20, 20)

    for i in range(M):
        if separating_function(theta, bias, D[i]) >= 0:
            labeled_dataset[i] = np.array([D[i, 0], D[i, 1], 1])

            if show_plot:
                plt.scatter(D[i, 0], D[i, 1], color="blue")

        else:
            labeled_dataset[i] = np.array([D[i, 0], D[i, 1], -1])

            if show_plot:
                plt.scatter(D[i, 0], D[i, 1], color="red")

    if show_plot:
        plt.plot(x_points, y_line_pos, color="green", linestyle="--", label="Separating Function")
        plt.plot(x_points, y_line_neg, color="green", linestyle="--")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(f"Dataset with Nonlinear Separating Function. Number of points: {M}")
        plt.legend()
        plt.grid(True)
        plt.show()

    return labeled_dataset


# Define the cost function and the gradient
def cost(theta, points):
    theta, bias = theta[:4], theta[4]

    cost = 0
    for i in range(len(points)):
        x = points[i, :2]
        p = points[i, 2]
        cost += np.log(1 + np.exp(-p * (np.dot(theta, phi(x)) + bias)))

    return cost


def cost_gradient(theta, points):
    theta, bias = theta[:4], theta[4]

    def sep_fn(x):
        return np.dot(theta, phi(x)) + bias

    gradient = np.zeros(5)
    for i in range(len(points)):
        x = points[i, :2]
        p = points[i, 2]

        gradient[0] += -p * x[0] * np.exp(-p * sep_fn(x)) / (1 + np.exp(-p * sep_fn(x)))
        gradient[1] += -p * x[1] * np.exp(-p * sep_fn(x)) / (1 + np.exp(-p * sep_fn(x)))
        gradient[2] += -p * x[0] ** 2 * np.exp(-p * sep_fn(x)) / (1 + np.exp(-p * sep_fn(x)))
        gradient[3] += -p * x[1] ** 2 * np.exp(-p * sep_fn(x)) / (1 + np.exp(-p * sep_fn(x)))
        gradient[4] += -p * np.exp(-p * sep_fn(x)) / (1 + np.exp(-p * sep_fn(x)))

    return gradient


def centralized_gradient(dataset):
    intermediate_results = []
    result = minimize(
        fun=cost,
        x0=np.zeros(5),
        args=(dataset,),
        method="BFGS",
        jac=cost_gradient,
        options={"disp": True},
        callback=lambda x: intermediate_results.append(x),
    )

    print(f"first result: {intermediate_results[0]}")

    # Plot evolution of the cost function

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    axs[0].plot([cost(theta, dataset) for theta in intermediate_results], color="blue")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Cost")
    axs[0].set_title("Evolution of the Cost Function")
    axs[0].grid(True)

    # Plot evolution of the norm of the gradient of the cost function
    axs[1].plot([np.linalg.norm(cost_gradient(theta, dataset)) for theta in intermediate_results], color="red")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Norm of the Gradient")
    axs[1].set_title("Evolution of the Norm of the Gradient")
    axs[1].grid(True)

    plt.tight_layout()

    plot_results(dataset, result.x, "Parameters found by centralized gradient")


def plot_results(dataset, theta, title):
    print(
        f"Parameters: a = {theta[0]:.2f}, b = {theta[1]:.2f}, c = {theta[2]:.2f}, d = {theta[3]:.2f}, bias = {theta[4]:.2f}, e = {np.sqrt(-theta[4]):.2f}"
    )

    w, bias = theta[:4], theta[4]

    # Plot the labeled dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset[dataset[:, 2] == 1, 0], dataset[dataset[:, 2] == 1, 1], color="blue")
    plt.scatter(dataset[dataset[:, 2] == -1, 0], dataset[dataset[:, 2] == -1, 1], color="red")

    # Plot the separating function as a line
    x_1 = np.linspace(-10, 10, 10000)

    y_line_pos = np.zeros(len(x_1))
    y_line_neg = np.zeros(len(x_1))

    for i in range(len(x_1)):
        y_line_pos[i] = ellipse_equation(w, bias, x_1[i])[0]
        y_line_neg[i] = ellipse_equation(w, bias, x_1[i])[1]

    plt.plot(x_1, y_line_pos, color="green", linestyle="--", label="Separating Function")
    plt.plot(x_1, y_line_neg, color="green", linestyle="--")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def classification_error(dataset, theta):
    w, bias = theta[:4], theta[4]

    error = 0
    for i in range(len(dataset)):
        x = dataset[i, :2]
        p = dataset[i, 2]

        if p * separating_function(w, bias, x) < 0:
            error += 1

    return error / len(dataset)
