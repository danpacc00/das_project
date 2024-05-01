#
# 22 Apr 2024
#
# Ivano Notarnicola
#
# NN
#
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
max_iters = 100


###############################################################################
# Activation Function
def loss_fn(xT, p):
    return np.linalg.norm(xT - p) ** 2, 2 * (xT - p)


# Activation Function
def sigmoid_fn(xi):
    return 1 / (1 + np.exp(-xi))


# Derivative of Activation Function
def activation_fn(xi):
    return sigmoid_fn(xi), sigmoid_fn(xi) * (1 - sigmoid_fn(xi))


# Inference: xtp = f(xt,ut)
def inference_dynamics(xt, ut):
    """
    input:
              xt current state (size=d)
              ut current input (size=d*(d+1)), including bias)
    output:
              xtp next state
    """
    d = xt.shape[0]
    xtp = np.zeros(d)

    for h in range(d):
        temp = xt @ ut[h, 1:] + ut[h, 0]  # because of the bias
        xtp[h], _ = activation_fn(temp)
    return xtp


# Forward Propagation
def forward_pass(uu, x0):
    """
    input:
              uu input trajectory: u[0],u[1],..., u[T-1]
              x0 initial condition
    output:
              xx state trajectory: x[1],x[2],..., x[T]
    """
    T = uu.shape[0] + 1
    d = x0.shape[0]
    xx = np.zeros((T, d))
    xx[0] = x0
    for t in range(T - 1):
        xx[t + 1] = inference_dynamics(xx[t], uu[t])

    return xx


# Adjoint dynamics:
def adjoint_dynamics(ltp, xt, ut):
    """
    input:
              llambda_tp current costate
              xt current state
              ut current input
    output:
              llambda_t next costate
              delta_ut loss gradient wrt u_t
    """
    d = xt.shape[0]
    df_dx = np.zeros((d, d))
    df_du = np.zeros(((d + 1) * d, d))
    idx = (d + 1) * np.ones((d), dtype=int)
    cs_idx = np.append(0, np.cumsum(idx))

    for h in range(d):
        temp = xt @ ut[h, 1:] + ut[h, 0]  # because of the bias
        _, disgma_h = activation_fn(temp)

        df_dx[:, h] = ut[h, 1:] * disgma_h
        df_du[cs_idx[h] : cs_idx[h + 1], h] = np.hstack([1, xt]) * disgma_h

    lt = df_dx @ ltp
    Delta_ut_vec = df_du @ ltp

    Delta_ut = np.reshape(Delta_ut_vec, (d, d + 1))
    return lt, Delta_ut


# Backward Propagation
def backward_pass(xx, uu, llambdaT):
    """
    input:
              xx state trajectory: x[1],x[2],..., x[T]
              uu input trajectory: u[0],u[1],..., u[T-1]
              llambdaT terminal condition
    output:
              llambda costate trajectory
              delta_u costate output, i.e., the loss gradient
    """
    T, d = xx.shape
    llambda = np.zeros(xx.shape)
    Delta_u = np.zeros((T - 1, d, d + 1))

    llambda[-1] = llambdaT

    for t in reversed(range(T - 1)):
        llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t + 1], xx[t], uu[t])

    return llambda, Delta_u


###############################################################################
# MAIN
###############################################################################

J = np.zeros(max_iters)  # Loss values
norm_grad_J = np.zeros(max_iters)  # gradient of the cost function

T = 5
d = 3
alpha = 1e-2

data_point = np.random.uniform(size=(d))
label_point = np.random.uniform(size=(d))

uu = np.zeros((T - 1, d, d + 1))
# GO!
for k in range(max_iters - 1):
    if ((k + 2) % 50) == 0:
        print(f"Cost at iter {k+2:4d} is {J[k-1]:.4f}")

    xx = forward_pass(uu, data_point)
    _, llambdaT = loss_fn(xx[-1], label_point)
    _, Delta_uu = backward_pass(xx, uu, llambdaT)
    uu -= alpha * Delta_uu

    J[k], _ = loss_fn(xx[-1], label_point)
    norm_grad_J[k] = np.linalg.norm(Delta_uu)

if 1:
    _, ax = plt.subplots()
    ax.plot(
        range(max_iters),
        J,
        marker=".",
        linestyle="--",
    )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$J(u^k)$")
    ax.grid()

plt.show()
