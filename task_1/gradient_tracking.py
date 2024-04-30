import networkx as nx
import numpy as np


class GradientTracking:
    def __init__(self, cost_fn, max_iters=1000, alpha=1e-2):
        self.cost_fn = cost_fn
        self.max_iters = max_iters
        self.alpha = alpha

        self.cost = np.zeros(max_iters)
        self.gradient_magnitude = np.zeros(max_iters)

    def run(self, graph):
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

        zz = np.zeros((self.max_iters, nn))
        ss = np.zeros((self.max_iters, nn))
        for ii in range(nn):
            _, ss[0, ii] = self.cost_fn(ii, zz[0, ii])

        for kk in range(1, self.max_iters):
            print(f"iter {kk}")

            for ii in range(nn):
                N_ii = np.nonzero(Adj[ii])[0]

                zz[kk, ii] += AA[ii, ii] * zz[kk - 1, ii]
                ss[kk, ii] += AA[ii, ii] * ss[kk - 1, ii]
                for jj in N_ii:
                    zz[kk, ii] += AA[ii, jj] * zz[kk - 1, jj]
                    ss[kk, ii] += AA[ii, jj] * ss[kk - 1, jj]

                zz[kk, ii] -= self.alpha * ss[kk - 1, ii]

                _, grad_ell_ii_new = self.cost_fn(ii, zz[kk, ii])

                self.gradient_magnitude[kk] += np.linalg.norm(grad_ell_ii_new)

                _, grad_ell_ii_old = self.cost_fn(ii, zz[kk - 1, ii])
                ss[kk, ii] += grad_ell_ii_new - grad_ell_ii_old

                ell_ii_gt, _ = self.cost_fn(ii, zz[kk, ii])
                self.cost[kk] += ell_ii_gt

        return zz, self.cost, self.gradient_magnitude
