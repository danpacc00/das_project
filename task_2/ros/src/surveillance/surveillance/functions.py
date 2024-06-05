import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

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


def animation(XX, horizon, Adj, targets):
    NN = XX.shape[1]

    for tt in range(len(horizon)):
        # plot trajectories
        plt.plot(
            XX[:, :, 0],
            XX[:, :, 1],
            color=gray_O4S,
            linestyle="dashed",
            alpha=0.5,
        )

        # plot targets
        for ii in range(targets.shape[0]):
            plt.plot(
                targets[ii, 0],
                targets[ii, 1],
                marker="x",
                markersize=15,
                fillstyle="full",
                color=blue_O4S,
            )

        # plot formation
        xx_tt = XX[tt, :, :]

        # add thebarycenter as a marker
        barycenter = np.mean(xx_tt, axis=0)
        plt.plot(
            barycenter[0],
            barycenter[1],
            marker="o",
            markersize=15,
            fillstyle="full",
            color=blue_O4S,
        )

        for ii in range(NN):
            p_prev = xx_tt[ii]

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
                    p_curr = xx_tt[jj]
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

        if tt == len(horizon) - 1:
            plt.pause(10)

        plt.clf()


def animation2(XX, horizon, Adj, targets, up_wall, down_wall, middle):
    NN = XX.shape[1]

    for tt in range(len(horizon)):
        # plot trajectories
        plt.plot(
            XX[:, :, 0],
            XX[:, :, 1],
            color=gray_O4S,
            linestyle="dashed",
            alpha=0.5,
        )

        # plot corridor
        plt.plot(up_wall[:, 0], up_wall[:, 1], "k")
        plt.plot(down_wall[:, 0], down_wall[:, 1], "k")
        plt.plot(
            middle[:, 0],
            middle[:, 1],
            linestyle="dashed",
            color=gray_O4S,
            alpha=0.2,
        )

        # plot targets
        for ii in range(targets.shape[0]):
            plt.plot(
                targets[ii, 0],
                targets[ii, 1],
                marker="x",
                markersize=15,
                fillstyle="full",
                color=blue_O4S,
            )

        # plot formation
        xx_tt = XX[tt, :, :]

        # add thebarycenter as a marker
        barycenter = np.mean(xx_tt, axis=0)
        plt.plot(
            barycenter[0],
            barycenter[1],
            marker="o",
            markersize=15,
            fillstyle="full",
            color=blue_O4S,
        )

        for ii in range(NN):
            p_prev = xx_tt[ii]

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
                    p_curr = xx_tt[jj]
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

        if tt == len(horizon) - 1:
            plt.pause(10)

        plt.clf()
