import matplotlib.pyplot as plt
import numpy as np


def ss_estimates(ss):
    _, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].semilogx(np.arange(ss.shape[0]), ss[:, :, 0])
    ax[0].grid()
    ax[0].set_title("$s_x$")
    ax[0].set_xlabel("Iterations (log scale)")
    ax[0].set_ylabel("Value")

    ax[1].semilogx(np.arange(ss.shape[0]), ss[:, :, 1])
    ax[1].grid()
    ax[1].set_title("$s_y$")
    ax[1].set_xlabel("Iterations (log scale)")
    ax[1].set_ylabel("Value")
    plt.suptitle("Estimates of the barycenter")

    plt.show()


def cost_gradient(cost, gradient_magnitude, title_suffix):
    _, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].semilogy(np.arange(len(cost)), cost)
    ax[0].grid()
    ax[0].set_title("Cost")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Value (log scale)")

    ax[1].semilogy(np.arange(len(gradient_magnitude)), gradient_magnitude)
    ax[1].grid()
    ax[1].set_title("Gradient magnitude")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Value (log scale)")

    plt.suptitle(f"Aggregative tracking {title_suffix}")
    plt.show()


def convergence(diff_barycenter_s, v_nabla2_diff):
    _, ax = plt.subplots(1, 2, figsize=(10, 10))
    # Plot difference between barycenter and s in log scale
    ax[0].semilogy(np.arange(diff_barycenter_s.shape[0]), diff_barycenter_s)
    ax[0].grid()
    ax[0].set_title("Difference between barycenter and s (estimation of the barycenter)")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("ss - barycenter (log scale)")

    ax[1].semilogy(np.arange(v_nabla2_diff.shape[0]), v_nabla2_diff)
    ax[1].grid()
    ax[1].set_title("Difference between $\\nabla_2\\ell(z, \\sigma(z))$ of the cost and vv (estimation)")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("$\\nabla_2\\ell(z, \\sigma(z))$ - vv (log scale)")

    plt.suptitle("Convergence")
    plt.show()


def trajectories(zz, targets, zz_init, case, tradeoff, additional_elements=[]):
    for jj in range(zz.shape[1]):
        plt.plot(
            zz[:, jj, 0],
            zz[:, jj, 1],
            linewidth=1,
            color="black",
            linestyle="dashed",
            label=f"Trajectory {jj}",
        )

        plt.scatter(zz[-1, jj, 0], zz[-1, jj, 1], color="orange", marker="x")
        plt.annotate(
            f"$z_{jj}^0$",
            xy=(zz[0, jj, 0], zz[0, jj, 1]),
            xytext=(zz[0, jj, 0] + 0.2, zz[0, jj, 1] + 0.2),
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="red"),
        )

        plt.plot(targets[:, 0], targets[:, 1], "bx")
        plt.plot(zz_init[:, 0], zz_init[:, 1], "ro")

        if case == 0:
            label_offsets = [(0.2, 0.2), (-0.2, -0.7), (-0.8, -0.7), (0.2, -0.2)]
        else:
            label_offsets = [(0.1, 0.2), (0.1, 0.2), (-0.55, -0.35), (-0.55, -0.35)]

        plt.annotate(
            f"Target {jj}",
            xy=(targets[jj, 0], targets[jj, 1]),
            xytext=(targets[jj, 0] + label_offsets[jj][0], targets[jj, 1] + label_offsets[jj][1]),
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="blue"),
        )

        plt.title(f"Agents trajectories ($\\gamma = {tradeoff}$)")

        print(f"Final distance from target node {jj}: ", np.linalg.norm(zz[-1, jj] - targets))

    for elem in additional_elements:
        elem()

    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-60, 60)
    plt.show()


def corridor(top_wall, bottom_wall, y_offset, x, g_1, g_2):
    # Plot the corridor walls
    plt.plot(
        np.linspace(top_wall["x_start"], top_wall["x_end"], top_wall["res"]),
        np.tile(top_wall["y"], top_wall["res"]),
        "k",
    )
    plt.plot(
        np.linspace(bottom_wall["x_start"], bottom_wall["x_end"], bottom_wall["res"]),
        np.tile(bottom_wall["y"], bottom_wall["res"]),
        "k",
    )
    plt.plot(
        np.tile(top_wall["x_start"], top_wall["res"]),
        np.linspace(top_wall["y"], top_wall["y"] + y_offset * 4, top_wall["res"]),
        "k",
    )
    plt.plot(
        np.tile(bottom_wall["x_start"], bottom_wall["res"]),
        np.linspace(bottom_wall["y"], bottom_wall["y"] - y_offset * 4, bottom_wall["res"]),
        "k",
    )
    plt.plot(
        np.tile(top_wall["x_end"], top_wall["res"]),
        np.linspace(top_wall["y"], top_wall["y"] + y_offset * 4, top_wall["res"]),
        "k",
    )
    plt.plot(
        np.tile(bottom_wall["x_end"], bottom_wall["res"]),
        np.linspace(bottom_wall["y"], bottom_wall["y"] - y_offset * 4, bottom_wall["res"]),
        "k",
    )

    # Plot the corridor barrier functions
    plt.plot(x, g_1, color="green", linestyle="dashed")
    plt.plot(x, g_2, color="green", linestyle="dashed")
