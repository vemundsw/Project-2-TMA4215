
import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.lax as lax

jax.config.update("jax_enable_x64", True)

def charged_particle_eq(t: float, y: jnp.array, B: callable, U_grad: callable) -> jnp.array: 

    """
    RHS of the governing equation of a particle moving in in a static magnetic field

    y: array on form (x1, x2, x2, v1, v2, v3)
    t: time 
    B: function describing magnetic field strength
    U_grad: function describing gradient of electrostatic potential

    Returns the derivative of y according to the equation
    """
    x1_dot = y[3]
    x2_dot = y[4]
    x3_dot = y[5]
    y_dot = jnp.cross(y[3:], B(y[:3])) - U_grad(y[:3])

    return jnp.concatenate((x1_dot, x2_dot, x3_dot, y_dot), axis = None)

# 2D charged particle 
x_0 = jnp.array([0, 1.0, 0.1])
v_0 = jnp.array([0.09, 0.05, 0.20])

def B(x: jnp.array) -> jnp.array:

    """
    Static magnetic field for 2D charged particles

    x: position on form (x1, x2, x3)

    Returns the magnetic field strength
    """

    return jnp.array([0, 0, jnp.sqrt(x[0] ** 2 + x[1] ** 2)])

@jax.jit
def A(x: jnp.array) -> jnp.array:

    """
    Vector potential for static magnetic field for 2D charged particle

    x: position on form (x1, x2, x3)
    
    Returns the vector potential
    """

    dist = jnp.sqrt(x[0] ** 2 + x[1] ** 2)

    return (1/3) * jnp.array([-x[1] * dist, x[0] * dist, 0 * dist])

@jax.jit
def U(x: jnp.array) -> float:
    """
    Electrostatic potential for 2D charged particle
    
    x: position on form (x1, x2, x3)
        
    Returns the potential
    """

    dist = jnp.sqrt(x[0] ** 2 + x[1] ** 2)

    return 1/ (100 * dist)

@jax.jit
def gradU(x :jnp.array) -> jnp.array:
    """
    Gradient of electrostatic potential for 2D charged particle
        
    x: position on form (x1, x2, x3)
            
    Returns the gradient
    """

    dist = jnp.sqrt(x[0] ** 2 + x[1] ** 2) ** 3

    return -1/100 * jnp.array([x[0] / dist, x[1]/ dist, 0])









def initial_velocity(x0: jnp.array, v0: jnp.array, h: float = 0.1) -> jnp.array:
    """
    Calculates the initial velocity v(t_0 - h/2).
    The Boris method uses positions and velocity at timeintervals h/2  apart.

    x0: x(t_0) array with shape 3.
    v0: v(t_0) arrau of shape 3.
    
    Returns v(t_0 -h/2)
    """

    # Using solve_ivp to calculate the v(-0.05)
    def system(t: jnp.array, Y: jnp.array, h: float = 0.1):
        x = Y[:3]
        v = Y[3:]
        dxdt = v
        dvdt = jnp.cross(v, B(x)) - gradU(x)
        return jnp.concatenate((dxdt, dvdt))

    Y0 = jnp.concatenate((x0, v0))    # Everything in solvebvp must be one in one vector.

    sol = solve_ivp(system, (0, -h/2), Y0)
    v_initial = sol.y[3:, -1]

    return v_initial



def Boris_step(x: jnp.array, v: jnp.array, h: jnp.array = 0.1) -> tuple[jnp.array, jnp.array]:
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
    v_prime = v_minus + jnp.cross(v_minus, t)
    v_plus = v_minus + jnp.cross(v_prime, s) 

    v_next = v_plus - h/2 * gradU(x)
    x_next = x + h * v_next

    return x_next, v_next



def Boris_method(x0: jnp.array, v0: jnp.array, T: int = 10**4, h: float = 0.1,) -> tuple[jnp.array, jnp.array]:
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

    x_values = jnp.zeros(shape=(N,3))
    x_values[0] = x0
    v_values = jnp.zeros(shape=(N,3))
    v_values[0] = v_init

    x = x0
    v = v_init

    def body(carry, _):
        x, v  = carry
        x, v = Boris_step(x,v)

        return (x, v), (x,v)

    (_, _), trajectory = lax.scan(body, (x0, v0), xs = None, length = N)


    return trajectory




