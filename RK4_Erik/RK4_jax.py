import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)


def RK4_step(t:jax.Array, Yn:jax.Array, system:jax.Array, h:float) ->jax.Array:
    """
    t: time (is ignored here)
    Yn: [x1,x2,x3,v1,v2,v3]
    system: gives dY/dt (RHS of differential equation)
    h: steplength

    returns: Yn_new

    """
    k1 = h * system(t, Yn)
    k2 = h * system(t, Yn + 0.5*k1)
    k3 = h * system(t, Yn + 0.5*k2)
    k4 = h * system(t, Yn + k3)

    Yn_new = Yn + 1/6*(k1 + 2*k2 + 2*k3 + k4)

    return Yn_new

def RK4(t:jax.Array, T: int,  Y0:jax.Array, system:jax.Array, h:float) ->jax.Array:
    """
    t: time (is ignored here)
    T: Time duration (how long it should run)
    Y0: initial Y vector
    system: RHS differential eqt
    h: step length
    """
    def body(Yn, _):
        Yn_new = RK4_step(t, Yn, system, h)
        return Yn_new, Yn_new

    N = int(T/h)
    times = 0 + h *jnp.arange(N + 1)

    Y_final, trajectory = lax.scan(body, Y0, xs=None, length=N)
    
    Y = jnp.concatenate((Y0[None, :], trajectory), axis=0).T

    return Y, times
