import numpy as np


def r(x: np.array) -> float:
    return np.sqrt(x[0]**2+ x[1]**2)

def B(x: np.array) -> np.array:
    return np.array([0, 0, r(x)])


def A(x: np.array) -> np.array:
    return 1/3 * np.array([-x[1]*r(x), x[0]*r(x), 0])


def U(x: np.array) -> float:
    rr = max(r(x), 1e-20) # TO make sure the I dont get divide by zero errors
    return 1 / (100 * rr)

def gradU(x: np.array) -> np.array:
    rr= max(r(x), 1e-20) # TO make sure the I dont get divide by zero errors
    return -1/ 100 *np.array([x[0]/rr**3, x[1]/rr**3, 0])


# # Initial conditions
# x0 = np.array([0, 1, 0.1])
# v0 = np.array([0.09, 0.05, 0.20])

# h = 0.1    # Steplength
# T = 10**5  # Total time 

# System of differential equations
def system(t: np.array, Y: np.array) -> np.array:
    x = Y[:3]
    v = Y[3:]
    dxdt = v
    dvdt = np.cross(v, B(x)) - gradU(x)
    return np.concatenate((dxdt, dvdt))