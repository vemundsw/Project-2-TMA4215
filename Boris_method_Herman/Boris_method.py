import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# import jax as jnp

def r(x):
    return np.sqrt(x[0]**2+ x[1]**2)

def B(x):
    return np.array([0, 0, r(x)])


def A(x):
    return 1/3 * np.array([-x[1]*r(x), x[0]*r(x), 0])


def U(x):
    rr = max(r(x), 1e-20) # TO make sure the I dont get divide by zero errors
    return 1 / (100 * rr)

def gradU(x):
    rr= max(r(x), 1e-20) # TO make sure the I dont get divide by zero errors
    return -1/ 100 *np.array([x[0]/rr**3, x[1]/rr**3, 0])


# Initial conditions
x0 = np.array([0, 1, 0.1])
v0 = np.array([0.09, 0.05, 0.20])

h = 0.1    # Steplength in time
T = 10**5  # Total time 
# Time array becomes (0, 0.1, 0.2, 0.3, ... , 99 999.8, 99 999.9, 100 000)


def initial_velocity(x0, v0):

    # Using solve_ivp to calculate the v(-0.05)
    def system(t, Y):
        x = Y[:3]
        v = Y[3:]
        dxdt = v
        dvdt = np.cross(v, B(x)) - gradU(x)
        return np.concatenate((dxdt, dvdt))

    Y0 = np.concatenate((x0, v0))    # Everything in solvebvp must be one in one vector.

    # # solve_ivp( f(tn, yn) = y', t_span, intitial value y0)
    # sol = solve_ivp(system, (0, -0.1), Y0)
    # v_m1 = sol.y[3:, -1]
    # v_initial = (v0 + v_m1)/2

    # Another way of finding the intitial value, without averaging.
    sol = solve_ivp(system, (0, -0.05), Y0)
    v_initial = sol.y[3:, -1]

    return v_initial




# Staggered grid. the velocities are half a grid point behind the position values
def Boris_step(x, v, h):
    v_minus = v -  h/2 * gradU(x)    #v_minus

    t = h/2 * B(x)
    s = 2 * t / (1 + t @ t)
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s) 

    v_next = v_plus - h/2 * gradU(x)
    x_next = x + h * v_next

    return x_next, v_next


def Boris_method():
    x0 = np.array([0, 1, 0.1])
    v0 = np.array([0.09, 0.05, 0.20])

    T = 10**4
    h= 0.1
    N = int(T/h)

    v_init = initial_velocity(x0, v0)

    x_values = np.zeros(shape=(N,3))
    x_values[0] = x0
    v_values = np.zeros(shape=(N,3))
    v_values[0] = v0


    ###########Fix listene her!

    x = x0
    v = v_init
    for i in range(N-1):
        x, v = Boris_step(x, v, h)
        x_values[i+1] = x
        v_values[i+1] = v

    return np.array(x_values), np.array(v_values)

positions, velocities = Boris_method()

radius = np.linalg.norm(positions[:, :2], axis=1)
plt.plot(radius)
print(h*np.max(radius))


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(positions[:,0], positions[:, 1], label = "Position")
ax.set_ylabel("Position y")
ax.set_xlabel("Position x")
ax.grid(True)

ax.legend()
# plt.savefig("boris.pdf", bbox_inches="tight")
plt.show()



# def energy(x, v):
#     return 1/2 * np.linalg.norm(v, 2)**2 + U(x)


# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(np.linspace(0, 10**4, 10**5), energy(positions, velocities), label = "energy")
# ax.set_ylabel("Energy over time")
# ax.set_xlabel("Time")
# ax.grid(True)

# ax.legend()
# # plt.savefig("boris.pdf", bbox_inches="tight")
# plt.show()

energy = (
    0.5*np.sum(velocities**2, axis=1)
    + np.array([U(x) for x in positions])
)

plt.figure()
plt.plot([n for n in range(len(energy))], energy)
plt.show()


#Next step:
# Check why the trajectory is not stable.
# Implement Jax JIT.