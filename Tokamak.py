import numpy as np


def r2(x):
    return x[0]**2+ x[1]**2

def r(x):
    return np.sqrt(x[0]**2+ x[1]**2)

def B(x):
    return np.array([-(2 * x[1] + x[0]*x[2]) / (2 *r2(x)),
                          (2*x[0] - x[1]*x[2]) / 2* r2(x),
                                    (r(x) -1) / (2* r(x))])


 def A(x):
    return np.array([                                         0,
                           1/2  * ( x[0]- np.log(r(x) + x[0]) ),
           -1/2 * (x[2] * np.arctan(x[1]/x[0]) + np.log(r2(x))) ])

def U(x):
    return 0

def gradU(x):
    return 0

x0 = np.array([1.2, 0, 0])
v0 = np.array([0, 4.816* 10**(-4),  -2.059*10**(-3)])
