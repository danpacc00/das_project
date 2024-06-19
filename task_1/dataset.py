import numpy as np


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

    discriminant = b**2 - 4 * d * (c * x**2 + a * x + bias)
    if discriminant < 0:
        return np.nan, np.nan

    y_positive = (-b + np.sqrt(discriminant)) / (2 * d)
    y_negative = (-b - np.sqrt(discriminant)) / (2 * d)

    return y_positive, y_negative


def create_labeled_dataset(params, M):
    a = params[0]  # Stretch wrt x-axis
    b = params[1]  # Tilt wrt x-axis
    c = params[2]  # Coefficient for x^2. Controls the width of the ellipse
    d = params[3]  # Coefficient for y^2. Controls the height of the ellipse
    e = params[4]

    w = np.array([a, b, c, d])  # Weights
    bias = -(e**2)  # Bias

    x_lim = 15
    y_lim = d

    offset = 1.5

    D_1 = np.random.uniform(-x_lim - offset, x_lim + offset, size=(M, 1))
    D_2 = np.random.uniform(-y_lim - offset, y_lim + offset, size=(M, 1))

    D = np.concatenate((D_1, D_2), axis=1)

    labeled_dataset = np.zeros((M, 3))

    for i in range(M):
        if separating_function(w, bias, D[i]) >= 0:
            labeled_dataset[i] = np.array([D[i, 0], D[i, 1], 1])
        else:
            labeled_dataset[i] = np.array([D[i, 0], D[i, 1], -1])

    return labeled_dataset


def cost(theta, points):
    w, bias = theta[:4], theta[4]

    cost = 0
    for i in range(len(points)):
        x = points[i, :2]
        p = points[i, 2]
        cost += np.log(1 + np.exp(-p * (np.dot(w, phi(x)) + bias)))

    return cost


def cost_gradient(theta, points):
    w, bias = theta[:4], theta[4]

    def sep_fn(x):
        return np.dot(w, phi(x)) + bias

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


def centralized_gradient(dataset, initial_theta=None, alpha=1e-4, max_iters=1000, d=5):
    costs = np.zeros(max_iters)
    gradient_magnitude = np.zeros(max_iters)

    theta = np.zeros(d)
    if initial_theta is not None:
        theta = initial_theta

    for ii in range(max_iters - 1):
        costs[ii] = cost(theta, dataset)
        gradient = cost_gradient(theta, dataset)
        theta -= alpha * gradient
        gradient_magnitude[ii] = np.linalg.norm(gradient)

        print(f"Iteration: #{ii}, Cost: {costs[ii]:.2f}, Gradient Magnitude: {gradient_magnitude[ii]:.2f}")

        if gradient_magnitude[ii] < 1e-2:
            break

    return theta, costs[: ii + 1], gradient_magnitude[: ii + 1]


def classification_error(dataset, theta):
    w, bias = theta[:4], theta[4]

    error = 0
    for i in range(len(dataset)):
        x = dataset[i, :2]
        p = dataset[i, 2]

        if p * separating_function(w, bias, x) < 0:
            error += 1

    return error / len(dataset)
