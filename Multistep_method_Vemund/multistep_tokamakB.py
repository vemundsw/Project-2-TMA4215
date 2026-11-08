# Herman: La til disse 4 linjene for å hente kode fra en annen mappe.
# Ser om det blir mer oversiktlig.
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from fourth_order_multistep_jax import fourth_order_sym_multistep_lax,  charged_particle_eq, create_grids
import matplotlib.pyplot as plt
from Differential_equations.Tokamak import B, r, gradU, A, r2, U
import scipy
import jax
import jax.numpy as jnp
import time
import numpy as np

#jax.config.update("jax_enable_x64", True)

T = 1E3
t_0 = 0
h = 0.1 
N = 7       # number of steplengths 

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

# def A(x):
#     return jnp.array([0 * r(x), 
#                      1/2  * ( x[0]- np.log(r(x) + x[0]) ), 
#                      -1/2 * (x[2] * np.arctan(x[1]/x[0]) + np.log(r2(x))) ])

def B(x):
    return np.array([-(2 * x[1] + x[0]*x[2]) / (2 *r2(x)),
                          (2*x[0] - x[1]*x[2]) / (2* r2(x)),
                                    (r(x) -1) / (2* r(x))])


def gradU_jax(x: jnp.array) -> jnp.array:
    return 0*x

def A_jax(x):
    return jnp.array([0 * r_jax(x), 
                     1/2  * ( x[0]- jnp.log(r_jax(x) + x[0]) ), 
                     -1/2 * (x[2] * jnp.arctan(x[1]/x[0]) + jnp.log(r2(x))) ])

def r_jax(x): 
    return jnp.sqrt(x[0]**2+ x[1]**2)

@jax.jit
def jacobian_tokamak(x: jnp.array) -> jnp.array:

    """
    Analytically computed Jacobian for A(x) in the Tokamak magnetic field

    x: position on form (x1, x2, x3)

    Returns a 3x3 Jacobian matrix
    """

    r = jnp.sqrt(x[0]**2+ x[1]**2)
    r2 = x[0]**2+ x[1]**2

    return jnp.array([ [0*r,     0*r     , 0*r],
                      [1/2 - (x[0]/r + 1) / (2 * (x[0] + r)), -x[1] / (2*(x[0] + r) * r), 0*r],
                       [-x[0]/r2 + x[1]*x[2]/(2 * r2), -x[1]/ r2 - x[2] / (2 * x[0] * (1 + (x[1]/ x[0])**2)), -jnp.atan(x[1]/ x[0]) / 2]])

# Initial conditions
x0 = jnp.array([1.2, 0, 0])
v0 = jnp.array([0, 4.816e-4,  -2.059*10e-3])


#results = fourth_order_sym_multistep_lax(x0, v0, alpha_list, h, beta0, beta1, t_0, T, B, gradU, A_jax)
x, v, t = create_grids(x0, v0, h, t_0, T, B, gradU)

# Create a reference solution
y_0 = jnp.concatenate((x0, v0), axis = None)
y_0 = np.asarray(y_0)
ref_sol = scipy.integrate.solve_ivp(charged_particle_eq, (t_0, T), y_0, args = (B, gradU), dense_output=True, method = "RK45", rtol=1e-10, atol=1e-12)


errors = []
h_list = []
h4_list = []
times = []
for i in range(N):

    h_list.append(h)
    h4_list.append(h**4)
    
    x, v, t = create_grids(x0, v0, h, t_0, T, B, gradU)
    x_ref = ref_sol.sol(t)[:3][:, :-1]
    v_ref = ref_sol.sol(t)[3:][:, :-1]

    start = time.perf_counter()
    results = fourth_order_sym_multistep_lax(x, v, int(t.size), alpha_list, h, beta0, beta1,  gradU_jax, A_jax, jacobian_tokamak)
    stop = time.perf_counter()
    print(f"With h = {h} and T = {T} the multistep method uses {round(stop - start, 4)}s")
    times.append(stop-start)

    errors.append( np.linalg.norm(np.asarray(results["x"]) - x_ref, ord = 2))

    h /= 2

# We expect order 4 so the error graph should be paralell to (h_i, h_i^4)
plt.plot(h_list, h4_list, label = r"$(h_i, h_i^4)$")
    
plt.plot(h_list, errors, label = r"error for $h_i$", marker = "o")
plt.loglog()
plt.xlabel("Steplength (h)")
plt.xlim(h_list[0] + 0.1 * h_list[0], h_list[-1] - 0.1 * h_list[-1])
plt.ylabel("Two norm error")
plt.title("Errors with decreasing steplengths")
plt.legend()
plt.show()

for i in range(N):
    plt.plot(times[i], errors[i],  marker = "o", label = f"h = {h_list[i]/0.1}h")

plt.loglog()
plt.xlabel("Time (s)")
plt.ylabel("Two norm error")
plt.title("Work-precision multistep Tokamok")
plt.legend()
plt.show()


