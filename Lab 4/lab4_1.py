import numpy as np


def grad(x):
    return np.array(
        [3 * np.exp(3 * x[0]) - 2 * x[2] * x[0], -4 * np.exp(-4 * x[1]) - 2 * x[2] * x[1], - x[0] ** 2 - x[1] ** 2 + 1])


def rh_gen(F):
    def rh_update(z, n):
        return -F(z)

    return rh_update


def hessian(x):
    row_0 = np.array([9 * np.exp(3 * x[0]) - 2 * x[2], 0, -2 * x[0]]).reshape(3, 1)
    row_1 = np.array([0, 16 * np.exp(-4 * x[1]) - 2 * x[2], -2 * x[1]]).reshape(3, 1)
    row_2 = np.array([-2 * x[0], -2 * x[1], 0]).reshape(3, 1)
    return np.concatenate((row_0, row_1, row_2), axis=1)


def hess_lag_gen(H):
    def hess_lag_update(z, n):
        return H(z)

    return hess_lag_update


def step(z, rh, hess_lag, rh_update, hess_lag_update, alpha=1, epsilon=1e-10):
    dz = np.linalg.solve(hess_lag, rh)
    # print(matrix,rh)
    z = z + alpha * dz
    rh = rh_update(z, n)
    hess_lag = hess_lag_update(z, n)
    eps_condition = np.linalg.norm(rh[:2]) > epsilon
    return eps_condition, z, rh, hess_lag


def loop(z, n):
    eps_condition = True
    niter = 1000
    i_loop = 0
    rh_update = rh_gen(grad)
    hess_lag_update = hess_lag_gen(hessian)
    rh = rh_update(z, n)
    hess_lag = hess_lag_update(z, n)
    while eps_condition and i_loop < niter:
        eps_condition, z, rh, hess_lag = step(z, rh, hess_lag, rh_update, hess_lag_update)
        i_loop += 1
    return z


if __name__ == '__main__':
    z0 = np.array([-1, 1, -1])
    n = "im not useful"
    print(loop(z0, n))
