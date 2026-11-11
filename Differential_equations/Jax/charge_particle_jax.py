import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

def r(x: jax.Array) -> float:
    return jnp.sqrt(x[0]**2+ x[1]**2)

def B(x: jax.Array) -> jax.Array:
    return jnp.array([0, 0, r(x)])


def A(x: jax.Array) -> jax.Array:
    return 1/3 * jnp.array([-x[1]*r(x), x[0]*r(x), 0])


def U(x: jax.Array) -> float:
    rr = jnp.clip(r(x), 1e-20, None) # To make sure the I dont get divide by zero errors
    return 1 / (100 * rr)

def gradU(x: jax.Array) -> jax.Array:
    rr= jnp.clip(r(x), 1e-20, None) # TO make sure the I dont get divide by zero errors
    return -1/ 100 *jnp.array([x[0]/rr**3, x[1]/rr**3, 0])


# # Initial conditions
# x0 = jnp.Array([0, 1, 0.1])
# v0 = jnp.Array([0.09, 0.05, 0.20])

# h = 0.1    # Steplength
# T = 10**5  # Total time 

# System of differential equations
@jax.jit
def system(t: jax.Array, Y: jax.Array) -> jax.Array:
    x = Y[:3]
    v = Y[3:]
    dxdt = v
    dvdt = jnp.cross(v, B(x)) - gradU(x)
    return jnp.concatenate((dxdt, dvdt))