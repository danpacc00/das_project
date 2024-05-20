from functions import animation, error_plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def formation_vector_field(x, n_x, distances, Adj):
    """dx_i = ... for all i"""

    NN = Adj.shape[0]
    XX_dot = np.zeros(x.shape)
    span_n_x = np.arange(n_x)

    for ii in range(NN):
        N_i = np.where(Adj[:, ii] > 0)[0]
        index_ii = ii * n_x + span_n_x
        x_i = x[index_ii]

        for jj in N_i:
            index_jj = jj * n_x + span_n_x
            x_j = x[index_jj]
            dVij = (np.linalg.norm(x_i - x_j) ** 2 - distances[ii, jj] ** 2) * (
                x_i - x_j
            )
            XX_dot[index_ii] -= dVij

    return XX_dot


NN = 6
n_x = 2
L = 2

distances = np.array(
    [
        [0, L, 0, 2 * L, 0, L],
        [L, 0, L, 0, 2 * L, 0],
        [0, L, 0, L, 0, 2 * L],
        [2 * L, 0, L, 0, L, 0],
        [0, 2 * L, 0, L, 0, L],
        [L, 0, 2 * L, 0, L, 0],
    ]
)


Adj = distances > 0  # bmatrix with boolean elements

print(distances)
print(Adj)

# Then we need to code the dynamics

Tmax = 3.0
XX_init = np.random.rand(n_x * NN)  # Random initial conditions for the agents

fc_dynamics = (
    lambda t, x: formation_vector_field(x, n_x, distances, Adj)
)  # lambda function to pass the parameters. Lambda is a way to define a function in one line.

# function of scipy, scipy.integrate. It takes the function that you want to integrate, the time span,
# and the initial conditions. The result is a dictionary with the solution of the ODE.
res = solve_ivp(
    fun=fc_dynamics,
    t_span=[0, Tmax],
    y0=XX_init,
)

horizon = res.t
XX = res.y


if 0:
    plt.figure("Distance error")
    error_plot(XX, NN, n_x, Adj, distances, horizon)

if 1:
    plt.figure("Animation")
    animation(XX, NN, n_x, horizon, Adj)

plt.show()
