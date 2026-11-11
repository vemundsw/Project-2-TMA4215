import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

def r2(x: jax.Array) -> float:
    return x[0]**2+ x[1]**2

def r(x: jax.Array) -> float:
    return jnp.sqrt(x[0]**2+ x[1]**2)

def B(x: jax.Array) -> jax.Array:
    """
    Returns the forces of the magnetic Field
    """

    rr2 = jnp.maximum(r2(x), 1e-20) # TO make sure the I dont get divide by zero errors
    rr = jnp.maximum(r(x), 1e-20) # TO make sure the I dont get divide by zero errors

    return jnp.array([-(2 * x[1] + x[0]*x[2]) / (2 *rr2),
                          (2*x[0] - x[1]*x[2]) / (2* rr2),
                                    (r(x) -1) / (2* rr)])


def A(x: jax.Array) -> jax.Array:

    rr2 = jnp.maximum(r2(x), 1e-20) # TO make sure the I dont get divide by zero errors
    rr = jnp.maximum(r(x), 1e-20) # TO make sure the I dont get divide by zero errors

    return jnp.array([0, 
                     1/2  * ( x[0] - jnp.log(rr + x[0]) ), 
                     -1/2 * (x[2] * jnp.arctan(x[1]/x[0]) + jnp.log(rr2)) ])

def U(x: jax.Array) -> float:
    return 0

def gradU(x: jax.Array) -> float:
    return 0


# # System of differential equations
@jax.jit
def system(t: float, Y: jax.Array) -> jax.Array:
    x = Y[:3]
    v = Y[3:]
    dxdt = v
    dvdt = jnp.cross(v, B(x)) - gradU(x)
    return jnp.concatenate((dxdt, dvdt))