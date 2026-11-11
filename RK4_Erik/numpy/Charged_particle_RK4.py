from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Differential_equations.charge_particle import system as system_CharPar
from Differential_equations.charge_particle import U
from RK4_Erik.RK4 import RK4
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import time



#5A
#1
t = np.array([0]) #can be ignored
T = 100
Y0 = np.array([0, 1, 0.1, 0.09, 0.05, 0.20])
h = 0.1

CharPar_sol, times = RK4(t, T, Y0, system_CharPar, h)

X = CharPar_sol[:3, :]
x1 = X[0, :]
x2 = X[1, :]

#plt.plot(x1, x2)
#plt.show()


#2
#find refrence first
solution = solve_ivp(system_CharPar, (0, T), Y0, method="RK45", t_eval=times, rtol=1e-12, atol=1e-12)
X_ref = solution.y[:3, :]

norm_compare = np.linalg.norm(X - X_ref, axis=0)

#plt.plot(times, norm_compare)
#plt.semilogy()
#plt.show()

#3
#energy convervation
V = CharPar_sol[3:, :]
E = 0.5 * np.power(np.linalg.norm(V, axis=0), 2) + U(X)

e_n = np.abs(E - E[0])

#plt.plot(times, e_n)
#plt.show()


#B
errors = np.zeros(3)
runtimes = np.zeros(3)
h_list = np.zeros(3)
X_Ni_ref = X_ref[:, -1]

for i in range(len(h_list)):
    hi = h/2**i
    h_list[i] = hi

    #RK4
    start = time.perf_counter()
    CharPar_sol_i, times_i = RK4(t, T, Y0, system_CharPar, hi)
    end = time.perf_counter()

    X_Ni_method = CharPar_sol_i[:3, -1]
    runtime = end - start

    errors[i] = np.linalg.norm(X_Ni_method - X_Ni_ref)
    runtimes[i] = runtime


#1
ref_order2 = h_list**2
ref_order4 = h_list**4

plt.loglog(h_list, errors, "o-", label = "RK4")
plt.loglog(h_list, ref_order2, "o-", label=r"$h^2$")
plt.loglog(h_list, ref_order4, "o-", label=r"$h^4$")
plt.xlabel("Steglengde h")
plt.ylabel("Posisjonsfeil ved sluttid")
plt.title("RK4")
plt.legend()
plt.show()

#2
plt.loglog(errors, runtimes, "o-", label="RK4")

for i in range(len(errors)):
    plt.annotate(
        f"h = {h_list[i]:g}",
        (errors[i], runtimes[i]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel(r"Sluttfeil $e_i$")
plt.ylabel("Kjøretid (s)")
plt.legend()
plt.show()
