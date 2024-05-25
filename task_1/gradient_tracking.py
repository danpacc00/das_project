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

        # Metropolis-Hastings algorithm
        for ii in range(nn):
            N_ii = np.nonzero(Adj[ii])[0]
            deg_ii = len(N_ii)
            for jj in N_ii:
                deg_jj = len(np.nonzero(Adj[jj])[0])
                AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))

        AA += np.eye(nn) - np.diag(np.sum(AA, axis=0))

        zz = np.zeros((self.max_iters, nn, d))
        # zz[0, :, :] = np.random.uniform(size=(nn, d))
        # zz[0, :, :] = np.array([9, 2, 1, 5, 0.5]) + np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        zz[0, :, :] = zz0
        ss = np.zeros((self.max_iters, nn, d))
        for ii in range(nn):
            _, ss[0, ii, :] = self.cost_fn(ii, zz[0, ii, :])

        for kk in range(1, self.max_iters):
            for ii in range(nn):
                N_ii = np.nonzero(Adj[ii])[0]

                zz[kk, ii, :] += AA[ii, ii] * zz[kk - 1, ii, :]
                ss[kk, ii, :] += AA[ii, ii] * ss[kk - 1, ii, :]
                for jj in N_ii:
                    zz[kk, ii, :] += AA[ii, jj] * zz[kk - 1, jj, :]
                    ss[kk, ii, :] += AA[ii, jj] * ss[kk - 1, jj, :]

                zz[kk, ii, :] -= self.alpha * ss[kk - 1, ii, :]

                _, grad_ell_ii_new = self.cost_fn(ii, zz[kk, ii, :])

                self.gradient_magnitude[kk] += np.linalg.norm(grad_ell_ii_new)

                _, grad_ell_ii_old = self.cost_fn(ii, zz[kk - 1, ii, :])
                ss[kk, ii, :] += grad_ell_ii_new - grad_ell_ii_old

                ell_ii_gt, _ = self.cost_fn(ii, zz[kk, ii, :])
                self.cost[kk] += ell_ii_gt

            print(f"Iteration: #{kk}, Cost: {self.cost[kk]:.2f}, Gradient Magnitude: {self.gradient_magnitude[kk]:.2f}")

            # if self.gradient_magnitude[kk] < 1e-6:
            #     print("Converged")
            #     break

            # Take the gradient magnitude from the last 10 iterations and check if it's not changing a lot, then stop
            if kk > 10 and np.std(self.gradient_magnitude[kk - 10 : kk]) < 1e-2:
                print("Converged")

                return zz[:kk, :, :], self.cost[:kk], self.gradient_magnitude[:kk]

        return zz[:kk, :, :], self.cost[:kk], self.gradient_magnitude[:kk]
