import numpy as np


def r2(x: np.array) -> float:
    return x[0]**2+ x[1]**2

def r(x: np.array) -> float:
    return np.sqrt(x[0]**2+ x[1]**2)

def B(x: np.array) -> np.array:
    """
    Returns the forces of the magnetic Field
    """

    rr2 = max(r2(x), 1e-20) # TO make sure the I dont get divide by zero errors
    rr = max(r(x), 1e-20) # TO make sure the I dont get divide by zero errors

    return np.array([-(2 * x[1] + x[0]*x[2]) / (2 *rr2),
                          (2*x[0] - x[1]*x[2]) / (2* rr2),
                                    (r(x) -1) / (2* rr)])


def A(x: np.array) -> np.array:

    rr2 = max(r2(x), 1e-20) # TO make sure the I dont get divide by zero errors
    rr = max(r(x), 1e-20) # TO make sure the I dont get divide by zero errors

    return np.array([0, 
                     1/2  * ( x[0] - np.log(rr + x[0]) ), 
                     -1/2 * (x[2] * np.arctan(x[1]/x[0]) + np.log(rr2)) ])

def U(x: np.array) -> float:
    return 0

def gradU(x: np.array) -> float:
    return 0



# # Initial conditions
# x0 = np.array([1.2, 0, 0])
# v0 = np.array([0, 4.816* 10**(-4),  -2.059*10**(-3)])

# h = 0.1    # Steplength
# T = 10**5  # Total time 

# # System of differential equations
def system(t: np.array, Y: np.array) -> np.array:
    x = Y[:3]
    v = Y[3:]
    dxdt = v
    dvdt = np.cross(v, B(x)) - gradU(x)
    return np.concatenate((dxdt, dvdt))