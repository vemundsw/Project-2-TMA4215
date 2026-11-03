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


# Staggered grid. the velocities are half a grid point behind the position values
def Boris_step(x, v, h):
    vm = v -  h/2 * gradU(x)    #v_minus

    t = h/2 * B(x)
    s = 2 * t / (1 + t @ t)
    v_prime = vm + np.cross(vm, t)
    vp = vm + v- + np.cross(v_prime, s) 

    v_next = vp - h/2 * gradU(x)
    x_next = x + h * v_next

    return x_next, v_next