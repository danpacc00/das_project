import numpy as np


def phi(x):
    # Assuming phi(x) is the feature transformation, e.g., identity for logistic regression.
    return np.array([x[0], x[1], x[0] ** 2, x[1] ** 2])


def cost_function(theta, points):
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
        exp_term = np.exp(-p * sep_fn(x))
        denom = 1 + exp_term

        gradient[0] += -p * x[0] * exp_term / denom
        gradient[1] += -p * x[1] * exp_term / denom
        gradient[2] += -p * x[0] ** 2 * exp_term / denom
        gradient[3] += -p * x[1] ** 2 * exp_term / denom
        gradient[4] += -p * exp_term / denom

    return gradient


def numerical_gradient(cost_func, theta, points, epsilon=1e-5):
    num_grad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)

    for i in range(len(theta)):
        perturb[i] = epsilon
        loss1 = cost_func(theta + perturb, points)
        loss2 = cost_func(theta - perturb, points)
        num_grad[i] = (loss1 - loss2) / (2 * epsilon)
        perturb[i] = 0  # Reset the perturbation vector

    return num_grad


# Example points array: Each row is [x1, x2, p]
points = np.array([[0.5, 1.2, 1], [1.0, -1.5, -1], [-0.3, 0.8, 1], [0.7, -0.2, -1]])

# Initial theta: [a, b, c, d, bias]
theta = np.array([0.1, -0.2, 0.3, -0.4, 0.5])

# Compute gradients
analytical_grad = cost_gradient(theta, points)
numerical_grad = numerical_gradient(cost_function, theta, points)

# Print both gradients to compare
print("Analytical Gradient:", analytical_grad)
print("Numerical Gradient:", numerical_grad)

# Compute the difference between the two gradients and if it's small enough, the gradients are correct.
diff = np.linalg.norm(analytical_grad - numerical_grad)
print("Difference between gradients:", diff)
print("Gradients are correct:", diff < 1e-5)
