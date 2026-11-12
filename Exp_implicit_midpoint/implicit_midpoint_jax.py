# Må bygge en stoooor matrise.
# Og så løse likningsystemet.
# Jobber med å bygge matrisen. Tunga må holdes rett i munn for det er mange kryssledd fra vektorproduktet.
# ER usikker på om jeg må ha N x N, 3N x3N eller 6N, x 6N matrise. Trolig det siste, siden vi må løse for hver av komponentene i hver vektor.

import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.lax as lax
import time


# Cp = 2d_charged partcle

@jax.jit
def Bcp(x: jnp.array) -> jnp.array:

    """
    Static magnetic field for 2D charged particles

    x: position on form (x1, x2, x3)

    Returns the magnetic field strength
    """

    return jnp.array([0, 0, jnp.sqrt(x[0] ** 2 + x[1] ** 2)])


@jax.jit
def Ucp(x: jnp.array) -> float:
    """
    Electrostatic potential for 2D charged particle
    
    x: position on form (x1, x2, x3)
        
    Returns the potential
    """

    dist = jnp.clip(jnp.sqrt(x[0] ** 2 + x[1] ** 2), 1e-15)

    return 1/ (100 * dist)

@jax.jit
def gradUcp(x :jnp.array) -> jnp.array:
    """
    Gradient of electrostatic potential for 2D charged particle
        
    x: position on form (x1, x2, x3)
            
    Returns the gradient
    """

    dist = jnp.clip(jnp.sqrt(x[0] ** 2 + x[1] ** 2) ** 3, 1e-15)

    return -1/100 * jnp.array([x[0] / dist, x[1]/ dist, 0])


def Energy(x: jnp.array, v: jnp.array, U = Ucp) -> float:

    """
    Energy as sum of kinetic energy and potential energy

    x: position on form (x1, x2, x3)
    v: velocity on form (v1, v2, v3)

    Returns energy as scalar
    """

    return 1/2 * jnp.linalg.norm(v, ord = 2, axis = 0) ** 2 + U(x)


def system(t: jnp.array, Y: jnp.array, h: float = 0.1):
    x = Y[:3]
    v = Y[3:]
    dxdt = v
    dvdt = jnp.cross(v, B(x)) - gradU(x)
    return jnp.concatenate((dxdt, dvdt))


def mid_point_step():


def build_matrix(x, v, T, h):

    
    N = T/h
    A = jnp.zeros((N, N))







def implicit_midpoint():
    # Have to build a matrix. 
