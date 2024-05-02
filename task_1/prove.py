import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate the dataset
M = 200  # Number of points
d = 2  # Dimension of data
q = 1  # Dimension of feature space (after transformation)

# Generate random points gaussian
D = np.random.rand(M, d) * 10

# Assign random labels (-1 or 1)
labels = np.random.choice([-1, 1], size=M)

# Step 2: Define the separating function
w = 1
b = 2  # Bias term


def phi(x):
    a = 0.5
    y = np.sqrt(1 - (x**2) / (a**2)) * 0.5
    return y


def separating_function(x):
    return w.dot(phi(x)) + b


# Step 3: Plot the dataset and separating function
plt.figure(figsize=(8, 6))

# Plot points with different colors based on labels
# plt.scatter(D[labels == -1], D[labels == -1], color="red", label="Class -1")
# plt.scatter(D[labels == 1], D[labels == 1], color="blue", label="Class 1")

# Plot the separating function as a line
x_1 = np.linspace(-10, 10, 10000)

y_line = np.zeros(len(x_1))

for i in range(len(x_1)):
    y_line[i] = w * phi((x_1[i])) + b


print("y_line", y_line)

plt.plot(y_line, color="green", linestyle="--", label="Separating Function")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dataset with Nonlinear Separating Function")
plt.legend()
plt.grid(True)
plt.show()
