from  fourth_order_multistep import fourth_order_sym_multistep, fourth_order_sym_multistep_step, charged_particle_eq, fourth_order_sym_multistep_jax
import matplotlib.pyplot as plt
import numpy as np
from Tokamak import B, r, gradU, A, r2, U
import scipy
import jax
import jax.numpy as jnp
T = 10**2
t_0 = 0
h = 0.1 

# Constants for fourth order symmetric multistep
a1 = -0.7
a2 = 0.1
a3 = 0.9

beta0 = (1/3) * (20*a1*a2*a3 - 4*(a1*a2 + a1*a3 + a2*a3) - 28*(a1 + a2 + a3) - 52)
beta1 = (1/3) * (2*a1*a2*a3 + 14*(a1*a2 + a1*a3 + a2*a3) + 26*(a1 + a2 + a3) + 38)

alpha_list = np.array([
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

def A(x):
    return np.array([0 * r(x), 
                     1/2  * ( x[0]- np.log(r(x) + x[0]) ), 
                     -1/2 * (x[2] * np.arctan(x[1]/x[0]) + np.log(r2(x))) ])
def A_jax(x):
    return jnp.array([0 * r_jax(x), 
                     1/2  * ( x[0]- jnp.log(r_jax(x) + x[0]) ), 
                     -1/2 * (x[2] * jnp.arctan(x[1]/x[0]) + jnp.log(r2(x))) ])

def r_jax(x): 
    return jnp.sqrt(x[0]**2+ x[1]**2)

# Initial conditions
x0 = np.array([1.2, 0, 0])
v0 = np.array([0, 4.816* 10**(-4),  -2.059*10**(-3)])

results = fourth_order_sym_multistep_jax(x0, v0, alpha_list, h, beta0, beta1, t_0, T, B, gradU, A_jax)

# Plotting x2 as function of x1
plt.plot(results["x"][0], results["x"][1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Particle in Tokamak field for fourth order symmetrical multistep method ")
plt.show()

# Plotting R as function of x3
R = r(results["x"])
plt.plot(results["x"][2], R)
plt.xlabel("x3")
plt.ylabel(r"$R= \sqrt{x^2 + y^2}$")
plt.title("Particle in Tokamak field for fourth order symmetrical multistep method ")
plt.show()


# Estimate error 
y_0 = np.concatenate((x0, v0), axis = None)
ref_sol = scipy.integrate.solve_ivp(charged_particle_eq, (t_0, T), y_0, args = (B, gradU), dense_output=True, method = "RK45", rtol=1e-10, atol=1e-12)
x_ref = ref_sol.sol(results["t"])[:3]
v_ref = ref_sol.sol(results["t"])[3:]
multistep_error = np.linalg.norm(results["x"] - x_ref, ord = 2, axis = 0)

plt.plot(results["t"], multistep_error)
plt.semilogy()
plt.xlabel("t")
plt.ylabel(r"2-norm error")
plt.title("Particle in Tokamak field for fourth order symmetrical multistep method ")
plt.show()

# Plot change in energy
def E(x: np.array, v: np.array) -> float:

    """
    Energy as sum of kinetic energy and potential energy

    x: position on form (x1, x2, x3)
    v: velocity on form (v1, v2, v3)

    Returns energy as scalar
    """

    return 1/2 * np.linalg.norm(v, ord = 2, axis = 0) ** 2 + U(x)

# Must exclude last two indices since v is computed by five point difference
energy_diff_multistep = abs(E(results["x"][:, :-2], results["v"][:, :-2]) - E(x0, v0))

plt.plot(results["t"][:-2], energy_diff_multistep)
plt.semilogy()
plt.xlabel("t")
plt.ylabel(r"$|E(x_{(n)}, v_{(n)}) - E(x_{(0)}, v_{(0)})|$")
plt.title("Energy of particle in Tokamak field")
plt.show()