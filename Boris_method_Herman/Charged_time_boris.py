import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.integrate import solve_ivp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Differential_equations.charge_particle import r, B, A, U, gradU, system
from Boris_method_Herman.Charged_particle_Boris_method import Boris_method, plot_Boris, Energy



# Initial conditions
x0 = np.array([0, 1, 0.1])
v0 = np.array([0.09, 0.05, 0.20])

T = 1E3
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
    y_0 = np.concatenate((x0, v0), axis = None)
    # t= np.arange(0, T+h/4, h)
    t= np.linspace(0, T, int(T/h))
    ref_sol = solve_ivp(system, (t_0, T), y_0, t_eval=t, method = "RK45", rtol=1e-8, atol=1e-10)
    x_ref = ref_sol.y[:3]
    v_ref = ref_sol.y[3:]
    x_ref = ref_sol.y[:3].T

    start = time.perf_counter()
    positions, velocitues = Boris_method(x0, v0, T, h)
    stop = time.perf_counter()
    print(f"With h = {h} and T = {T} the Boris method uses {round(stop - start, 4)}s")
    times.append(stop-start)

    global_errors.append(np.linalg.norm(positions[-1] - x_ref[-1], ord = 2))

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


