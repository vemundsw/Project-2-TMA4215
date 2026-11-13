# Siden magnetfeltet ikke er lineært, så blir dette ikke et lineært likningssystem.
# Vi løser det for hvert tidssteg av gangen, og bruker en residualmatrise.
import scipy
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial




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

    dist = jnp.maximum(jnp.sqrt(x[0] ** 2 + x[1] ** 2), 1e-15)

    return 1/ (100 * dist)

@jax.jit
def gradUcp(x :jnp.array) -> jnp.array:
    """
    Gradient of electrostatic potential for 2D charged particle
        
    x: position on form (x1, x2, x3)
            
    Returns the gradient
    """

    dist = jnp.maximum((x[0] ** 2 + x[1] ** 2)** 1.5, 1e-15)

    return -1/100 * jnp.array([x[0] / dist, x[1]/ dist, 0])


def Residual_matrix(z, x_n, v_n, h):
    """
    Builds a residual matrix. 
    We will use the Newton method, to solve the linear matrix equation.


    z: A guess for x_(n+1), y_(n+1)
    x_n: Our previous solution for x
    y_n: Our previous solution for y.
    h: Steplength in time

    returns the calcuulated residual matrix. THis is what we want to minimize.
    """

    x_new = z[:3]
    v_new = z[3:]

    x_mid = 0.5 * (x_n + x_new)
    v_mid = 0.5 * (v_n + v_new)

    F_x = x_new - x_n - h/2 * (v_n + v_new)

    F_v = v_new - v_n - h * (jnp.cross(v_mid, Bcp(x_mid)) - gradUcp(x_mid))

    return jnp.concatenate((F_x, F_v))

@jax.jit
def Newton_Solver(initial_guess, x_n, v_n, h, tol):
    """
    Solves the linear system for the given timestep.
    Implemented to use jax.jit and jax.lax.scan() todecrease compute times.

    initial_guess: Our guess for the next positions and velocities. 
    The guess is either the position and velocity from the last timestep
    or we calculate a guess using the Euler method.

    x_n: Previous position
    v_n: Previous velocity
    h: steplenght

    tol: tolerance

    return: 

    """

    max_iter = 100

    def condition(state):
        k, z, error = state
        return (error > tol) & (k < max_iter)


    def iteration(state):
        k, z, error = state
        z = initial_guess
        F = Residual_matrix(z, x_n, v_n, h)

        Jacobian = jax.jacfwd(Residual_matrix, argnums = 0)(z, x_n, v_n, h)   # Argnums = 0 makes it so we only differentiate z.

        dz = jnp.linalg.solve(Jacobian, -F)

        z_new = z + dz

        error_new = jnp.linalg.norm(Residual_matrix(z_new, x_n, v_n, h))

        return k + 1, z_new , error_new 

    F0 = Residual_matrix(initial_guess, x_n, v_n, h)

    intitial_state = (0, initial_guess, jnp.linalg.norm(F0))

    k, z, error = jax.lax.while_loop(condition, iteration, intitial_state)

    return z[:3], z[3:], error, k


@jax.jit
def Euler_guess(x_n, v_n, h):
    """
    Optional, makes a more accurate guess for the next position and velocity.

    """

    x_guess = x_n + h * v_n
    v_guess = v_n + h * (jnp.cross(v_n, Bcp(x_n)) - gradUcp(x_n))
    return jnp.concatenate((x_guess, v_guess))
    

@partial(jax.jit, static_argnames=("N",)) #Jax neess N to be static.
def implicit_midpoint(x0, v0, h, N, tol):

    def step(carry, _): 
        # Carry holds the the current x_n and v_n: Jax needs this variable to properly use lax.scan, which speeds up the process a lot.
        x_n, v_n = carry

        z = jnp.concatenate((x_n, v_n))    # The initial guess for the positions and velocities are the last postions and velocities. 
        # z = Euler_guess(x_n, v_n, h)     # makes a more accurate guess of z.

        x_new, v_new, error, k = Newton_Solver(z, x_n, v_n, h, tol)

        new_carry = (x_new, v_new)

        output = (x_new, v_new, error, k)

        return new_carry, output

    initial_carry = x0, v0 

    # Final carry contains the position and velocity, error and timestepcount for the last timestep. 
    # results conteins all the postions and velocities, errors and timestepcounts for all timestep, except the initital ones.
    final_carry, results = jax.lax.scan(step, initial_carry, xs = None, length = N)

    x_values, v_values, errors, iterations = results

    # Add the initial values.
    positions = jnp.concatenate ((x0[None, :], x_values))
    velocities = jnp.concatenate((v0[None, :], v_values))

    return positions, velocities, errors, iterations



x0 = jnp.array([0, 1, 0.1])
v0 = jnp.array([0.09, 0.05, 0.20])

h = 0.1    # Steplength in time
t_0 = 0
T = 10**5  # Total time
N = int(T/h)
tol = 1e-5

p, v, errors, iterations = implicit_midpoint(x0, v0, h, N, tol)


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(p[:,0], p[:, 1], label = "Position")
ax.set_ylabel("Position y")
ax.set_xlabel("Position x")
ax.grid(True)
ax.legend()
ax.set_title("Position in the 2D-charged particle system. Implicit midpoint method.")
plt.show()

print("Maximum residual:", jnp.max(errors))
print("Maximum Newton iterations:", jnp.max(iterations))
non_converges =jnp.sum(errors > tol)
print("Number of steps that dont converge:", non_converges)















