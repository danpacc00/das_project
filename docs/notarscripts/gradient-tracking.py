import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
NN = 10

I_NN = np.eye(NN)
while 1:
    Adj = np.random.binomial(n=1, p=0.3, size=(NN, NN))
    Adj = np.logical_or(Adj, Adj.T)
    Adj = np.logical_and(Adj, np.logical_not(I_NN)).astype(int)

    test = np.linalg.matrix_power(I_NN + Adj, NN)
    if np.all(test > 0):
        break
    # else:
    # continue


AA = np.zeros(shape=(NN, NN))

for ii in range(NN):
    N_ii = np.nonzero(Adj[ii])[0]
    deg_ii = len(N_ii)
    for jj in N_ii:
        deg_jj = len(np.nonzero(Adj[jj])[0])
        AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))

AA += I_NN - np.diag(np.sum(AA, axis=0))

if 0:
    print(np.sum(AA, axis=0))
    print(np.sum(AA, axis=1))


def quadratic_fn(z, q, r):
    return 0.5 * q * z * z + r * z, q * z + r


Q = np.random.uniform(size=(NN))
R = np.random.uniform(size=(NN))

MAXITERS = 1000
# dd = 3
ZZ = np.zeros((MAXITERS, NN))
cost = np.zeros((MAXITERS))

ZZ_gt = np.zeros((MAXITERS, NN))
SS_gt = np.zeros((MAXITERS, NN))
for ii in range(NN):
    _, SS_gt[0, ii] = quadratic_fn(ZZ_gt[0, ii], Q[ii], R[ii])

cost_gt = np.zeros((MAXITERS))

alpha = 1e-2

for kk in range(MAXITERS - 1):
    print(f"iter {kk}")

    # Distributed gradient
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]

        ZZ[kk + 1, ii] += AA[ii, ii] * ZZ[kk, ii]
        for jj in N_ii:
            ZZ[kk + 1, ii] += AA[ii, jj] * ZZ[kk, jj]

        _, grad_ell_ii = quadratic_fn(ZZ[kk + 1, ii], Q[ii], R[ii])

        ZZ[kk + 1, ii] -= alpha / (kk + 1) * grad_ell_ii

        ell_ii, _ = quadratic_fn(ZZ[kk, ii], Q[ii], R[ii])
        cost[kk] += ell_ii

    # gradient tracking
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]

        ZZ_gt[kk + 1, ii] += AA[ii, ii] * ZZ_gt[kk, ii]
        SS_gt[kk + 1, ii] += AA[ii, ii] * SS_gt[kk, ii]
        for jj in N_ii:
            ZZ_gt[kk + 1, ii] += AA[ii, jj] * ZZ_gt[kk, jj]
            SS_gt[kk + 1, ii] += AA[ii, jj] * SS_gt[kk, jj]

        ZZ_gt[kk + 1, ii] -= alpha * SS_gt[kk, ii]

        # print(Q[ii])
        _, grad_ell_ii_new = quadratic_fn(ZZ_gt[kk + 1, ii], Q[ii], R[ii])
        _, grad_ell_ii_old = quadratic_fn(ZZ_gt[kk, ii], Q[ii], R[ii])
        SS_gt[kk + 1, ii] += grad_ell_ii_new - grad_ell_ii_old

        ell_ii_gt, _ = quadratic_fn(ZZ_gt[kk, ii], Q[ii], R[ii])
        cost_gt[kk] += ell_ii_gt


fig, ax = plt.subplots()
ax.plot(np.arange(MAXITERS), ZZ)
ax.grid()
ax.set_title("Distributed gradient")

fig, ax = plt.subplots()
ax.plot(np.arange(MAXITERS), ZZ_gt)
ax.grid()
ax.set_title("Gradient tracking")


ZZ_opt = -np.sum(R) / np.sum(Q)
opt_cost = 0.5 * np.sum(Q) * ZZ_opt**2 + np.sum(R) * ZZ_opt
print(opt_cost)
print(cost[-2])
print(cost_gt[-2])

fig, ax = plt.subplots()
ax.semilogy(np.arange(MAXITERS - 1), np.abs(cost[:-1] - opt_cost))
ax.semilogy(np.arange(MAXITERS - 1), np.abs(cost_gt[:-1] - opt_cost))
ax.grid()
ax.legend(["Distributed gradient", "Gradient tracking"])
ax.set_title("Cost error")

plt.show()
