import numpy as np


def r2(x: np.array) -> float:
    return x[0]**2+ x[1]**2

def r(x: np.array) -> float:
    return np.sqrt(x[0]**2+ x[1]**2)

def B(x: np.array) -> np.array:
    """
    Returns the forces of the magnetic Field
    """

    return np.array([-(2 * x[1] + x[0]*x[2]) / (2 *r2(x)),
                          (2*x[0] - x[1]*x[2]) / 2* r2(x),
                                    (r(x) -1) / (2* r(x))])


def A(x: np.array) -> np.array:
    return np.array([0, 
                     1/2  * ( x[0]- np.log(r(x) + x[0]) ), 
                     -1/2 * (x[2] * np.arctan(x[1]/x[0]) + np.log(r2(x))) ])

def U(x: np.array) -> float:
    return 0

def gradU(x: np.array) -> float:
    return 0



# Initial conditions
x0 = np.array([1.2, 0, 0])
v0 = np.array([0, 4.816* 10**(-4),  -2.059*10**(-3)])

h = 0.1    # Steplength
T = 10**5  # Total time 

# # System of differential equations
# def system(t, Y):
#     x = Y[:3]
#     v = Y[3:]
#     dxdt = v
#     dvdt = np.cross(v, B(x)) - gradU(x)
#     return np.concatenate((dxdt, dvdt))

# Y0 = np.concatenate((x0, v0))    # Everything in solvebvp must be one in 