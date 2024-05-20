#
# Animations script
# Lorenzo Pichierri
# Bologna, 08/04/2024
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

blue_O4S = mcolors.to_rgb((0 / 255, 41 / 255, 69 / 255))
emph_O4S = mcolors.to_rgb((0 / 255, 93 / 255, 137 / 255))
red_O4S = mcolors.to_rgb((127 / 255, 0 / 255, 0 / 255))
gray_O4S = mcolors.to_rgb((112 / 255, 112 / 255, 112 / 255))


def dist_error(XX, NN, n_x, Adj, distances, horizon):
    TT = len(horizon)
    err = np.zeros((distances.shape[0], distances.shape[1], TT))

    for tt in range(TT):
        for ii in range(NN):
            N_ii = np.where(Adj[:, ii] > 0)[0]
            index_ii = ii * n_x + np.arange(n_x)
            XX_ii = XX[index_ii, tt]

            for jj in N_ii:
                index_jj = jj * n_x + np.arange(n_x)
                XX_jj = XX[index_jj, tt]
                norm_ij = np.linalg.norm(XX_ii - XX_jj)

                # relative error
                err[ii, jj, tt] = distances[ii, jj] - norm_ij
    return err


def error_plot(XX, NN, n_x, Adj, distances, horizon):
    # Evaluate the distance error
    err = dist_error(XX, NN, n_x, Adj, distances, horizon)
    dist_err = np.reshape(err, (NN * NN, np.size(horizon)))

    # generate figure
    for h in range(NN * NN):
        plt.plot(horizon, dist_err[h])

    plt.title("Agents distance error [m]")
    plt.yscale("log")
    plt.xlabel("$t$")
    plt.ylabel("$\|x_i^t-x_j^t\|-d_{ij}, i = 1,...,N$")
    plt.grid()


def animation(XX, NN, n_x, horizon, Adj):
    for tt in range(len(horizon)):
        # plot trajectories
        plt.plot(
            XX[0 : n_x * NN : n_x].T,
            XX[1 : n_x * NN : n_x].T,
            color=gray_O4S,
            linestyle="dashed",
            alpha=0.5,
        )

        # plot formation
        xx_tt = XX[:, tt].T
        for ii in range(NN):
            index_ii = ii * n_x + np.arange(n_x)
            p_prev = xx_tt[index_ii]

            plt.plot(
                p_prev[0],
                p_prev[1],
                marker="o",
                markersize=15,
                fillstyle="full",
                color=red_O4S,
            )

            for jj in range(NN):
                if Adj[ii, jj] & (jj > ii):
                    index_jj = (jj % NN) * n_x + np.arange(n_x)
                    p_curr = xx_tt[index_jj]
                    plt.plot(
                        [p_prev[0], p_curr[0]],
                        [p_prev[1], p_curr[1]],
                        linewidth=1,
                        color=emph_O4S,
                        linestyle="solid",
                    )

        axes_lim = (np.min(XX) - 1, np.max(XX) + 1)
        plt.xlim(axes_lim)
        plt.ylim(axes_lim)
        plt.axis("equal")
        plt.xlabel("first component")
        plt.ylabel("second component")
        plt.title(f"Formation Control - Simulation time = {horizon[tt]:.2f} s")
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()
