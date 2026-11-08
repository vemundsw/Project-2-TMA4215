import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Differential_equations.Tokamak import r, r2, B, A, U, gradU
from Boris_method_Herman.Boris_method_Tokamak import Boris_method



def charged_particle_eq(t: np.array, Y: np.array, h: float = 0.1):
    x = Y[:3]
    v = Y[3:]
    dxdt = v
    dvdt = np.cross(v, B(x)) - gradU(x)
    return np.concatenate((dxdt, dvdt))



# Initial conditions
x0 = np.array([1.2, 0, 0])
v0 = np.array([0, 4.816* 10**(-4),  -2.059*10**(-3)])

h = 0.1    # Steplength
t_0 = 0
T = 10**4  # Total time 
t = np.arange(t_0, T, h)


positions, velocities = Boris_method(x0, v0, T)

# Creating a reference solution to estimate error
y_0 = np.concatenate((x0, v0), axis = None)

# Lowered the tolerances from what was speciefied in the task, as this took a long time
ref_sol = solve_ivp(charged_particle_eq, (t_0, T), y_0, t_eval=t, method = "RK45", rtol=1e-8, atol=1e-10)


x_ref = ref_sol.y[:3]
v_ref = ref_sol.y[3:]

x_ref = ref_sol.y[:3].T

multistep_error = np.linalg.norm(positions - x_ref, ord = 2, axis = 1)

print(multistep_error.shape)

plt.plot(t, multistep_error)
plt.semilogy()
plt.xscale("log")
plt.xlabel("t")
plt.ylabel(r"2-norm error")
plt.title("Error of Boris method for the Tokamak system")
plt.show()