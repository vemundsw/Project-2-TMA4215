import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Differential_equations.Tokamak import r, r2, B, A, U, gradU, system
from Tokamak_Boris_method import Boris_method, plot_Boris, Energy



# Initial conditions
x0 = np.array([1.2, 0, 0])
v0 = np.array([0, 4.816* 10**(-4),  -2.059*10**(-3)])

h = 0.1    # Steplength
t_0 = 0
T = 10**5  # Total time 
t = np.arange(t_0, T, h)


positions, velocities = Boris_method(x0, v0, T)
plot_Boris(positions)




# Creating a reference solution to estimate error
y_0 = np.concatenate((x0, v0), axis = None)

# Lowered the tolerances from what was speciefied in the task, as this took a long time
ref_sol = solve_ivp(system, (t_0, T), y_0, t_eval=t, method = "RK45", rtol=1e-8, atol=1e-10)


x_ref = ref_sol.y[:3]
v_ref = ref_sol.y[3:]

x_ref = ref_sol.y[:3].T

Boris_error = np.linalg.norm(positions - x_ref, ord = 2, axis = 1)


plt.plot(t, Boris_error)
plt.semilogy()
plt.xscale("log")
plt.xlabel("t")
plt.ylabel(r"2-norm error")
plt.title("Error of Boris method for the Tokamak system")
plt.show()



energy_diff_Boris = abs(Energy(positions, velocities) - Energy(x0, v0))
plt.plot(t, Boris_error)
plt.semilogy()
plt.xscale("log")
plt.xlabel("t")
plt.ylabel(r"2-norm error")
plt.title("Energy difference of Boris method for the Tokamak system")
plt.show()

# Feilen og energien er nesten helt lik.

print("Done")