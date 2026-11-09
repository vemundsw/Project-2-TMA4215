import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.lax as lax
import time

jax.config.update("jax_enable_x64", True)


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


# System of differential equations
@jax.jit
def system(t: jnp.array, Y: jnp.array) -> jnp.array:
    x = Y[:3]
    v = Y[3:]
    dxdt = v
    dvdt = jnp.cross(v, B(x)) - gradU(x)
    return jnp.concatenate((dxdt, dvdt))

def Energy(x: jnp.array, v: jnp.array) -> float:

    """
    Energy as sum of kinetic energy and potential energy

    x: position on form (x1, x2, x3)
    v: velocity on form (v1, v2, v3)

    Returns energy as scalar
    """

    return 1/2 * jnp.linalg.norm(v, ord = 2, axis = 0) ** 2 + U(x)


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


@jax.jit
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


def plot_Boris(positions):

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(positions[:,0], positions[:, 1], label = "Position")
    ax.set_ylabel("Position y")
    ax.set_xlabel("Position x")
    ax.grid(True)
    ax.legend()
    ax.set_title("Position of a electrically charged particle")
    plt.show()
    return

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
    x = x0
    v = v_init

    def body(carry, _):
        x, v  = carry
        x, v = Boris_step(x,v)

        return (x, v), (x,v)

    (_, _), trajectory = lax.scan(body, (x, v), xs = None, length = N)

    positions, velocities = trajectory

    positions = jnp.concatenate([x0[None, :], positions], axis=0)

    velocities = jnp.concatenate([v_init[None, :], velocities], axis=0)
    v_averages = 1/2 * (velocities[:-1] + velocities[1:])

    return positions[:-1], v_averages


# Initial conditions
x0 = jnp.array([1.2, 0, 0])
v0 = jnp.array([0, 4.816* 10**(-4),  -2.059*10**(-3)])

h = 0.1    # Steplength
t_0 = 0
T = 10**4  # Total time 
t = jnp.arange(t_0, T, h)


positions, velocities = Boris_method(x0, v0, T)
print(f"Shape of positions = {positions.shape}, shape of velocities: {velocities.shape}")
print(f"positions= {positions[0]} velocities = {velocities[0]}")

plot_Boris(positions)


h = 0.1    # Steplength in time
t_0 = 0
T = 10**4  # Total time
t = jnp.arange(t_0, T, h)

positions, velocities = Boris_method(x0, v0, T)

def plot_error_and_energy(positions):
    print("Solving initial value problem to estimate the error.")
    # Creating a reference solution to estimate error
    y_0 = jnp.concatenate((x0, v0), axis = None)

    # Lowered the tolerances from what was speciefied in the task, as this took a long time
    ref_sol = solve_ivp(system, (t_0, T), y_0, t_eval=t, method = "RK45", rtol=1e-8, atol=1e-10)
    x_ref = ref_sol.y[:3]
    v_ref = ref_sol.y[3:]

    x_ref = ref_sol.y[:3].T

    Boris_error = jnp.linalg.norm(positions - x_ref, ord = 2, axis = 1)

    plt.plot(t, Boris_error)
    plt.semilogy()
    plt.xscale("log")
    plt.xlabel("t")
    plt.ylabel(r"2-norm error")
    plt.title("Error of Boris method for 2D charged particle")
    plt.show()

    energy_diff_Boris = abs(Energy(positions, velocities) - Energy(x0, v0))
    plt.plot(t, Boris_error)
    plt.semilogy()
    plt.xscale("log")
    plt.xlabel("t")
    plt.ylabel(r"2-norm error")
    plt.title("Energy difference of Boris method for the 2d_charged particle system")
    plt.show()

plot_error_and_energy(positions)


T = 1E2
t_0 = 0
h = 0.1 
N = 7       # number of steplengths 


global_errors = [] 
h_list = []
h2_list = []
# h4_list = []
times = []
for i in range(N):

    h_list.append(h)
    h2_list.append(h**2)
    # h4_list.append(h**4)
    
    # Creating a reference solution to estimate error
    y_0 = jnp.concatenate((x0, v0), axis = None)
    t= jnp.linspace(0, T, int(T/h))
    ref_sol = solve_ivp(system, (t_0, T), y_0, t_eval=t, method = "RK45", rtol=1e-8, atol=1e-10)
    x_ref = ref_sol.y[:3]
    v_ref = ref_sol.y[3:]
    x_ref = ref_sol.y[:3].T

    start = time.perf_counter()
    positions, velocities = Boris_method(x0, v0, T, h)
    stop = time.perf_counter()
    print(f"With h = {h} and T = {T} the Boris method uses {round(stop - start, 4)}s")
    times.append(stop-start)

    global_errors.append(jnp.linalg.norm(positions[-1] - x_ref[-1], ord = 2))

    print("positions shape:", positions.shape)
    print("x_ref shape:", x_ref.shape)


    print("h =", h)
    print("len(t) =", len(t))
    print("ref final time =", ref_sol.t[-1])
    print("boris steps =", len(positions))

    h /= 2

print(global_errors)

plt.plot(h_list, h2_list, label = r"$(h_i, h_i^2)$")
# plt.plot(h_list, h4_list, label = r"$(h_i, h_i^4)$")
    
plt.plot(h_list, global_errors, label = r"error for $h_i$", marker = "o")
plt.loglog()
plt.xlabel("Steplength (h)")
plt.xlim(h_list[0] + 0.1 * h_list[0], h_list[-1] - 0.1 * h_list[-1])
plt.ylabel("Two norm error")
plt.title("Errors with decreasing steplengths")
plt.legend()
plt.show()

for i in range(N):
    plt.plot(times[i], global_errors[i],  marker = "o", label = f"h = {h_list[i]/0.1}h")

plt.loglog()
plt.xlabel("Time (s)")
plt.ylabel("Two norm error")
plt.title("Work-precision Boris Charged particle")
plt.legend()
plt.show()


