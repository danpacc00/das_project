import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate the dataset
M = 200  # Number of points
d = 2  # Dimension of data
q = 1  # Dimension of feature space (after transformation)


# Assign random labels (-1 or 1)
labels = np.random.choice([-1, 1], size=M)

# Step 2: Define the separating function
a = 9.0  # Stretch wrt x-axis
b = 2.0  # Tilt wrt x-axis
c = 1.0  # Coefficient for x^2. Controls the width of the ellipse
d = 5  # Coefficient for y^2. Controls the height of the ellipse
e = 0.5  # Distance from center of ellispse to one of its foci. Determines the size.

bias = -(e**2)

w = np.array([a, b, c, d])  # Weights


def phi(x):
    x = np.array(x)
    x_new = np.hstack((x, x**2))
    return x_new


def separating_function(x):
    print("w shape", w.shape)
    print("phi shape", phi(x).shape)
    return np.dot(w, phi(x)) + bias


def ellipse_equation(x):
    """Ellipse equation: dy^2 +by +cx^2 +ax = e^2

    solution: y1/2 = (-b ± sqrt(b^2 - 4(dx(cx+a))) - e^2) / 2d"""

    y_positive = (-b + np.sqrt(b**2 - 4 * d * (c * x**2 + a * x - e**2))) / (2 * d)
    y_negative = (-b - np.sqrt(b**2 - 4 * d * (c * x**2 + a * x - e**2))) / (2 * d)

    return y_positive, y_negative


def create_labeled_dataset():
    # Step 3: Plot the dataset and separating function
    plt.figure(figsize=(8, 6))

    # Plot the separating function as a line
    x_1 = np.linspace(-10, 10, 10000)

    y_line_pos = np.zeros(len(x_1))
    y_line_neg = np.zeros(len(x_1))

    for i in range(len(x_1)):
        y_line_pos[i] = ellipse_equation(x_1[i])[0]
        y_line_neg[i] = ellipse_equation(x_1[i])[1]

    plt.plot(
        x_1, y_line_pos, color="green", linestyle="--", label="Separating Function"
    )
    plt.plot(x_1, y_line_neg, color="green", linestyle="--")

    axes = plt.gca()
    Xlim = axes.get_xlim()
    Ylim = axes.get_ylim()
    offset = 1.5

    D_1 = np.random.uniform(Xlim[0] - offset, Xlim[1] + offset, size=(M, 1))
    D_2 = np.random.uniform(Ylim[0] - offset, Ylim[1] + offset, size=(M, 1))

    D = np.concatenate((D_1, D_2), axis=1)

    # Plot the dataset not labeled
    # plt.scatter(D[:, 0], D[:, 1], color="black")

    # Label the data with the separating function
    labeled_dataset = np.zeros((M, 3))

    for i in range(M):
        if separating_function(D[i]) >= 0:
            labeled_dataset[i] = np.array([D[i, 0], D[i, 1], 1])
            plt.scatter(D[i, 0], D[i, 1], color="blue")

        else:
            labeled_dataset[i] = np.array([D[i, 0], D[i, 1], -1])
            plt.scatter(D[i, 0], D[i, 1], color="red")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Dataset with Nonlinear Separating Function")
    plt.legend()
    plt.grid(True)
    plt.show()

    return labeled_dataset


def logistic_regression_cost_fn(x):
    label = x[-1]
    return np.log(1 + np.exp(-label * separating_function(x)))


def classify_points():
    return w


classify_points()
