import numpy as np
from scipy.integrate import solve_ivp

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


# Initial conditions
x0 = np.array([0, 1, 0.1])
v0 = np.array([0.09, 0.05, 0.20])

h = 0.1    # Steplength
T = 10**5  # Total time 


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



# Using solve_ivp to calculate the v(-0.05)
def system(t, Y):
    x = Y[:3]
    v = Y[3:]
    dxdt = v
    dvdt = np.cross(v, B(x)) - gradU(x)
    return np.concatenate((dxdt, dvdt))

Y0 = np.concatenate((x0, v0))    # Everything in solvebvp must be one in one vector.


# solve_ivp( f(tn, yn) = y', t_span, intitial value y0)
sol = solve_ivp(system, (0, -0.1), Y0)

v_m1 = sol.y[3:, -1]

v_initial = (v0 + v_m1)/2
print(v_initial)


sol = solve_ivp(system, (0, -0.05), Y0)
v_m1 = sol.y[3:, -1]
print(v_m1)