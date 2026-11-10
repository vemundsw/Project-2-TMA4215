import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Differential_equations.Tokamak import r, r2, B, A, U, gradU




# # Initial conditions
# x0 = np.array([1.2, 0, 0])
# v0 = np.array([0, 4.816* 10**(-4),  -2.059*10**(-3)])

# h = 0.1    # Steplength
# T = 10**4  # Total time 


def initial_velocity(x0: np.array, v0: np.array, h: float = 0.1) -> np.array:
    """
    Calculates the initial velocity v(t_0 - h/2).
    The Boris method uses positions and velocity at timeintervals h/2  apart.

    x0: x(t_0) array with shape 3.
    v0: v(t_0) arrau of shape 3.
    
    Returns v(t_0 -h/2)
    """

    # Using solve_ivp to calculate the v(-0.05)
    def system(t: np.array, Y: np.array, h: float = 0.1):
        x = Y[:3]
        v = Y[3:]
        dxdt = v
        dvdt = np.cross(v, B(x)) - gradU(x)
        return np.concatenate((dxdt, dvdt))

    Y0 = np.concatenate((x0, v0))    # Everything in solvebvp must be one in one vector.

    sol = solve_ivp(system, (0, -h/2), Y0)
    v_initial = sol.y[3:, -1]

    return v_initial




# Staggered grid. the velocities are half a grid point behind the position values
def Boris_step(x: np.array, v: np.array, h: np.array = 0.1) -> tuple[np.array, np.array]:
    """
    Calculates one step of the Boris method


    x: Current position, array of size 3.
    v: Current velocity, array of size 3.
    h:  Steplength in time.
    
    return: x_next, v_nect
    """


    v_minus = v -  h/2 * gradU(x)    #v_minus

    t = h/2 * B(x)
    s = 2 * t / (1 + t @ t)
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s) 

    v_next = v_plus - h/2 * gradU(x)
    x_next = x + h * v_next

    return x_next, v_next


def Boris_method(x0: np.array, v0: np.array, T: int = 10**5, h: float = 0.1,) -> tuple[np.array, np.array]:
    """
    Boris_method, calculates the position and velocity of the particle at time t.

    x0: Iinitial position, array of size 3.
    v0: Iinitial velocity, array of size 3.
    T:  Total time.
    h:  Steplength in time.

    return: arrays of positions and velocities.
    """

    N = int(T/h)

    v_init = initial_velocity(x0, v0)

    x_values = np.zeros(shape=(N,3))
    x_values[0] = x0
    v_values = np.zeros(shape=(N,3))
    v_values[0] = v_init

    x = x0
    v = v_init
    for i in range(N-1):
        x, v = Boris_step(x, v, h)
        x_values[i+1] = x
        v_values[i+1] = v

    return np.array(x_values), np.array(v_values)




# positions, velocities = Boris_method(x0, v0, 10**5)


def plot_Boris(positions):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(positions[:,0], positions[:, 1], label = "Position")
    ax.set_ylabel("Position y")
    ax.set_xlabel("Position x")
    ax.grid(True)
    ax.legend()
    ax.set_title("Position of a electrically charged particle in the Tokamak system")
    plt.show()


    

    plt.figure()
    radius = np.sqrt(positions[:,0]**2 + positions[:,1]**2)
    plt.plot(radius, positions[:,2])
    plt.title("Radius of the particle curve")
    plt.show()

    return




def Energy(x: np.array, v: np.array) -> float:

    """
    Energy as sum of kinetic energy and potential energy

    x: position on form (x1, x2, x3)
    v: velocity on form (v1, v2, v3)

    Returns energy as scalar
    """

    return 1/2 * np.linalg.norm(v, ord = 2, axis = 0) ** 2 + U(x)







