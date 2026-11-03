import numpy as np

def r(x):
    return np.sqrt(x[0]**2+ x[1]**2)

def B(x):
    return np.array([0, 0, r(x)])


def A(x):
    return 1/3 * np.array([-x[1]*r(x), x[0]*r(x), 0])


def U(x):
    return 1 / (100 * r(x))

def gradU(x):
    return -1/ 100 *np.array([x[0]/r(x)**3, x[1]/r(x)**3, 0])

x0 = np.array([0, 1, 0.1])
v0 = np.array([0.09, 0.05, 0.20])