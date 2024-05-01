import numpy as np
import matplotlib.pyplot as plt


def nonlinear_function(x):
    # Define your nonlinear function here
    # This is a simple example where the function squares each element of the input
    return np.power(x, 2)


# Test the function
n_points = 100  # Number of points to test
d = 3  # Dimension of the domain
q = 2  # Dimension of the codomain

# Create a random vector of n_points points in the domain
x = np.random.rand(n_points, d)

# Apply the nonlinear function
y = nonlinear_function(x)

print("Input (domain):", x)
print("Output (codomain):", y)

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(x, y, "o")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
plt.show()
