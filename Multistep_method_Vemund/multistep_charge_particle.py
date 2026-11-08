from  fourth_order_multistep import fourth_order_sym_multistep, fourth_order_sym_multistep_step
import matplotlib.pyplot as plt
import numpy as np

T = 10**5
t_0 = 0
h = 0.1 
# 2D charged particle 
x_0 = np.array([0, 1.0, 0.1])
v_0 = np.array([0.09, 0.05, 0.20])

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
def charged_particle_eq(t: float, y: np.array, B: callable, U_grad: callable) -> np.array: 

    """
    RHS of the governing equation of a particle moving in in a static magnetic field

    y: array on form (x1, x2, x2, v1, v2, v3)
    t: time 
    B: function describing magnetic field strength
    U_grad: function describing gradient of electrostatic potential

    Returns the derivative of y according to the equation
    """
    x1_dot = y[3]
    x2_dot = y[4]
    x3_dot = y[5]
    y_dot = np.cross(y[3:], B(y[:3])) - U_grad(y[:3])

    return np.concatenate((x1_dot, x2_dot, x3_dot, y_dot), axis = None)



def B_charged_particle(x: np.array) -> np.array:

    """
    Static magnetic field for 2D charged particles

    x: position on form (x1, x2, x3)

    Returns the magnetic field strength
    """

    return np.array([0, 0, np.sqrt(x[0] ** 2 + x[1] ** 2)])

def A_charged_particle(x: np.array) -> np.array:

    """
    Vector potential for static magnetic field for 2D charged particle

    x: position on form (x1, x2, x3)
    
    Returns the vector potential
    """

    dist = np.sqrt(x[0] ** 2 + x[1] ** 2)

    return (1/3) * np.array([-x[1] * dist, x[0] * dist, 0 * dist])

def U_charged_particle(x: np.array) -> float:
    """
    Electrostatic potential for 2D charged particle
    
    x: position on form (x1, x2, x3)
        
    Returns the potential
    """

    dist = np.sqrt(x[0] ** 2 + x[1] ** 2)

    return 1/ (100 * dist)

def U_grad_charged_particle(x :np.array) -> np.array:
    """
    Gradient of electrostatic potential for 2D charged particle
        
    x: position on form (x1, x2, x3)
            
    Returns the gradient
    """

    dist = np.sqrt(x[0] ** 2 + x[1] ** 2) ** 3

    return -1/100 * np.array([x[0] / dist, x[1]/ dist, 0])


results = fourth_order_sym_multistep(x_0, v_0, alpha_list, h, beta0, beta1, t_0, T, B_charged_particle, U_grad_charged_particle, A_charged_particle)

plt.plot(results["x"][0], results["x"][1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("2D charged particle with fourth order symmetrical multitep method ")
plt.show()

print(results["x"])