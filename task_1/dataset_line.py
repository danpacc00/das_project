import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Step 1: Generate the dataset
M = 1000  # Number of points
d = 2  # Dimension of data


# Step 2: Define the separating function
def phi(x):
    return np.array(x)


def separating_function(w, x):
    return np.dot(w, phi(x))


def line_equation(a, b, x):
    return a * x + b


def create_labeled_dataset(show_plot=False):
    a = 1.0  # Slope of the separating line
    b = 0.0  # Intercept of the separating line

    # Step 3: Plot the dataset and separating function
    plt.figure(figsize=(8, 6))

    # Plot the separating function as a line
    x_1 = np.linspace(-10, 10, 10000)
    y_line = line_equation(a, b, x_1)
    plt.plot(x_1, y_line, color="green", linestyle="--", label="Separating Function")

    D_1 = np.random.uniform(-10, 10, size=(M, 1))
    D_2 = np.random.uniform(-10, 10, size=(M, 1))

    D = np.concatenate((D_1, D_2), axis=1)

    # Label the data with the separating function
    labeled_dataset = np.zeros((M, 3))

    for i in range(M):
        if separating_function(np.array([a, -1]), D[i]) + b >= 0:
            labeled_dataset[i] = np.array([D[i, 0], D[i, 1], 1])
            plt.scatter(D[i, 0], D[i, 1], color="blue")
        else:
            labeled_dataset[i] = np.array([D[i, 0], D[i, 1], -1])
            plt.scatter(D[i, 0], D[i, 1], color="red")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Dataset with Linear Separating Function")
    plt.legend()
    plt.grid(True)

    if show_plot:
        plt.show()

    return labeled_dataset


# Define the cost function and the gradient
def cost(theta, points):
    cost = 0
    for i in range(len(points)):
        x = points[i, :2]
        p = points[i, 2]
        cost += np.log(1 + np.exp(-p * (np.dot(theta, phi(x)))))
    return cost


def cost_gradient(theta, points):
    def sep_fn(x):
        return np.dot(theta, phi(x))

    gradient = np.zeros(2)
    for i in range(len(points)):
        x = points[i, :2]
        p = points[i, 2]
        exp_term = np.exp(-p * sep_fn(x))
        gradient += -p * x * exp_term / (1 + exp_term)
    return gradient


def classify_points(dataset):
    intermediate_results = []
    result = minimize(
        fun=cost,
        x0=np.zeros(2),
        args=(dataset,),
        method="BFGS",
        jac=cost_gradient,
        options={"disp": True},
        callback=lambda x: intermediate_results.append(x),
    )

    print(f"first result: {intermediate_results[0]}")

    # Plot evolution of the cost function
    plt.figure(figsize=(8, 6))
    plt.plot([cost(theta, dataset) for theta in intermediate_results], color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Evolution of the Cost Function")
    plt.grid(True)
    plt.show()

    # Plot evolution of the norm of the gradient of the cost function
    plt.figure(figsize=(8, 6))
    plt.plot([np.linalg.norm(cost_gradient(theta, dataset)) for theta in intermediate_results], color="red")
    plt.xlabel("Iteration")
    plt.ylabel("Norm of the Gradient")
    plt.title("Evolution of the Norm of the Gradient")
    plt.grid(True)
    plt.show()

    plot_results(dataset, result.x)


def plot_results(dataset, theta):
    # Plot the labeled dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset[dataset[:, 2] == 1, 0], dataset[dataset[:, 2] == 1, 1], color="blue")
    plt.scatter(dataset[dataset[:, 2] == -1, 0], dataset[dataset[:, 2] == -1, 1], color="red")

    # Plot the separating function as a line
    x_1 = np.linspace(-10, 10, 10000)
    y_line = line_equation(-theta[0] / theta[1], 0, x_1)

    plt.plot(x_1, y_line, color="green", linestyle="--", label="Separating Function")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Dataset with Linear Separating Function")
    plt.legend()
    plt.grid(True)
    plt.show()
