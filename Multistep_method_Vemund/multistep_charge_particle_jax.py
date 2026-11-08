from fourth_order_multistep_jax import  fourth_order_sym_multistep_step_fast, create_grids, fourth_order_sym_multistep_lax
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import time 

jax.config.update("jax_enable_x64", True)

T = 10**5
t_0 = 0
h = 0.1 
# 2D charged particle 
x_0 = jnp.array([0, 1.0, 0.1])
v_0 = jnp.array([0.09, 0.05, 0.20])

# Constants for fourth order symmetric multistep
a1 = -0.7
a2 = 0.1
a3 = 0.9

beta0 = (1/3) * (20*a1*a2*a3 - 4*(a1*a2 + a1*a3 + a2*a3) - 28*(a1 + a2 + a3) - 52)
beta1 = (1/3) * (2*a1*a2*a3 + 14*(a1*a2 + a1*a3 + a2*a3) + 26*(a1 + a2 + a3) + 38)

alpha_list = jnp.array([
    1,           # α₋₄
    -7/5,        # α₋₃
    9/25,        # α₋₂
    22/125,      # α₋₁
    -34/125,     # α₀
    22/125,      # α₁
    9/25,        # α₂
    -7/5,        # α₃
    1            # α₄
])

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

@jax.jit
def A_charged_jacobian(x: jnp.array) -> jnp.array:

    r = jnp.sqrt(x[0]**2 + x[1]**2)

    return jnp.array([[-1/3*x[0]*x[1]/ r, -1/3 * (r + x[1]**2/ r), 0*r],
                        [1/3 * (r + x[0]**2/r), 1/3 * x[0] * x[1]/r, 0*r],
                        [0*r,                   0*r,                      0*r]])


def B_charged_particle(x: jnp.array) -> jnp.array:

    """
    Static magnetic field for 2D charged particles

    x: position on form (x1, x2, x3)

    Returns the magnetic field strength
    """

    return jnp.array([0, 0, jnp.sqrt(x[0] ** 2 + x[1] ** 2)])

@jax.jit
def A_charged_particle(x: jnp.array) -> np.array:

    """
    Vector potential for static magnetic field for 2D charged particle

    x: position on form (x1, x2, x3)
    
    Returns the vector potential
    """

    dist = jnp.sqrt(x[0] ** 2 + x[1] ** 2)

    return (1/3) * jnp.array([-x[1] * dist, x[0] * dist, 0 * dist])

@jax.jit
def U_charged_particle(x: jnp.array) -> float:
    """
    Electrostatic potential for 2D charged particle
    
    x: position on form (x1, x2, x3)
        
    Returns the potential
    """

    dist = jnp.sqrt(x[0] ** 2 + x[1] ** 2)

    return 1/ (100 * dist)

@jax.jit
def U_grad_charged_particle(x :jnp.array) -> jnp.array:
    """
    Gradient of electrostatic potential for 2D charged particle
        
    x: position on form (x1, x2, x3)
            
    Returns the gradient
    """

    dist = jnp.sqrt(x[0] ** 2 + x[1] ** 2) ** 3

    return -1/100 * jnp.array([x[0] / dist, x[1]/ dist, 0 * dist])


#results = fourth_order_sym_multistep_jax(x_0, v_0, alpha_list, h, beta0, beta1, t_0, T, B_charged_particle, U_grad_charged_particle, A_charged_particle)

x, v, t = create_grids(x_0, v_0, h, t_0, T, B_charged_particle, U_grad_charged_particle)
closure_U_grad = jax.tree_util.Partial(U_grad_charged_particle)
closure_A = jax.tree_util.Partial(A_charged_particle)
closure_jacobian = jax.tree_util.Partial(A_charged_jacobian)

start = time.time()
results = fourth_order_sym_multistep_lax(x, v, int(t.size), alpha_list, h, beta0, beta1, U_grad_charged_particle, A_charged_particle, A_charged_jacobian)
stop = time.time()
print(f"With T = {T} the jitted multistep takes {round(stop - start, 4)}s")

plt.plot(results["x"][0], results["x"][1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("2D charged particle with fourth order symmetrical multitep method ")
plt.show()

#print(results["x"])