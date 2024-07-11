import networkx as nx
import numpy as np


class GradientTracking:
    def __init__(self, cost_fn, max_iters=1000, alpha=1e-2):
        self.cost_fn = cost_fn
        self.max_iters = max_iters
        self.alpha = alpha

        self.cost = np.zeros(max_iters)
        self.gradient_magnitude = np.zeros(max_iters)

    def run(self, graph, d, zz0):
        nn = len(nx.nodes(graph))
        Adj = nx.adjacency_matrix(graph).toarray()

        AA = np.zeros(shape=(nn, nn))

        grad_s_diff = np.zeros((self.max_iters, nn))

        # Metropolis-Hastings weights in order to have a doubly stochastic matrix
        for ii in range(nn):
            N_ii = np.nonzero(Adj[ii])[0]
            deg_ii = len(N_ii)
            for jj in N_ii:
                deg_jj = len(np.nonzero(Adj[jj])[0])
                AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))

        AA += np.eye(nn) - np.diag(np.sum(AA, axis=0))

        zz = np.zeros((self.max_iters, nn, d))
        zz[0, :, :] = zz0

        ss = np.zeros((self.max_iters, nn, d))
        for ii in range(nn):
            _, ss[0, ii, :] = self.cost_fn(ii, zz[0, ii, :])

        for kk in range(self.max_iters - 1):
            grad = np.zeros(d)
            for ii in range(nn):
                N_ii = np.nonzero(Adj[ii])[0]

                zz[kk + 1, ii, :] += AA[ii, ii] * zz[kk, ii, :]
                ss[kk + 1, ii, :] += AA[ii, ii] * ss[kk, ii, :]
                for jj in N_ii:
                    zz[kk + 1, ii, :] += AA[ii, jj] * zz[kk, jj, :]
                    ss[kk + 1, ii, :] += AA[ii, jj] * ss[kk, jj, :]

                zz[kk + 1, ii, :] -= self.alpha * ss[kk, ii, :]

                _, grad_ell_ii_new = self.cost_fn(ii, zz[kk + 1, ii, :])

                _, grad_ell_ii_old = self.cost_fn(ii, zz[kk, ii, :])
                ss[kk + 1, ii, :] += grad_ell_ii_new - grad_ell_ii_old

                ell_ii_gt, _ = self.cost_fn(ii, zz[kk, ii, :])
                self.cost[kk] += ell_ii_gt
                grad += grad_ell_ii_new

            self.gradient_magnitude[kk] += np.linalg.norm(grad)

            for jj in range(nn):
                grad_s_diff[kk, jj] = np.linalg.norm(ss[kk, jj, :] - grad, axis=0)

            print(f"Iteration: #{kk}, Cost: {self.cost[kk]:.2f}, Gradient Magnitude: {self.gradient_magnitude[kk]:.2f}")

            if self.gradient_magnitude[kk] < 1e-6:
                print("Converged")
                break

        return zz[:kk, :, :], self.cost[:kk], self.gradient_magnitude[:kk], grad_s_diff[:kk, :]
