import numpy as np
import scipy
from scipy.differentiate import jacobian
import matplotlib.pyplot as plt

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

# 2D charged particle 
x_0 = np.array([0, 1.0, 0.1])
v_0 = np.array([0.09, 0.05, 0.20])

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


def delta_yj(y: np.array) -> float:
    """
    Five point central difference centered in y[2]

    y: array on form (y_j-2, y_j-1, y_j, y_j+1, y_j+2). Each y has 3 components

    Returns the central difference
    """

    return 1/12 * (y[:,0] - 8 * y[:,1] + 8* y[:,3] - y[:,4]) 

def F_j(A :callable, x :np.array,  h: float, U_grad: callable) -> np.array:
    """
    Fourth order symmetric multistep method help function

    x: array with shape (x_j-2, x_j-1, x_j, x_j+1, x_j+2). Each x has 3 compontents
    h: steplength
    U_grad: potential gradient function

    Returns a 3 component array
    """

    x_j = x[:, 2]
    delta_Aj = delta_yj(A(x))/ h
    vj = delta_yj(x)/ h
    A_jacobi = jacobian(A, x_j)

    return A_jacobi.df.T @ vj - delta_Aj - U_grad(x_j)

def fourth_order_sym_multistep_step(x: np.array, alpha: np.array, h: float, beta0: float, beta1: float, F: tuple) -> np.array:

    """
    Takes a step in the fourth order symmetrical multistep method

    x: array on form (x_n-4, x_n-3, ... , x_n+4)
    alpha: array on form (alpha_n-4, alpha_n-3, ... , alpha_n+4)
    h: steplength
    beta0, beta1: method parameters, 
    F: tuple on form (F_n-1, F_n, F_n+1)

    Returns an array x with updated x_n+4
    """

    prev_steps = np.sum(x[:,:-1] * alpha[:-1], axis = 1)
    x_n4 = 1/ alpha[-1] *( h**2 * (beta1 * F[0] + beta0 * F[1] + beta1 * F[2]) - prev_steps)
    #x[-1] = x_n4

    return x_n4


def fourth_order_sym_multistep(x_0: np.array, v_0: np.array, alpha: np.array, h: float, beta0: float, beta1: float, t_0: float, t_f: float, B: callable, U_grad: callable, A: callable) -> np.array:

    """
    Takes a step int the fourth order symmetrical multistep method

    x: array on form (x_n-4, x_n-3, ... , x_n+4)
    alpha: array on form (alpha_n-4, alpha_n-3, ... , alpha_n+4)
    h: steplength
    beta0, beta1: method parameters, 
    F: array on form (F_n-1, F_n, F_n+1)
    U_grad: potential gradient function
    A: magnetic field vector potential function

    Returns an array x with updated x_n+4
    """

    # Create grids
    t = np.arange(t_0, t_f, h)
    x = np.zeros(( 3, t.size))
    v = np.zeros(( 3, t.size))

    # Find first eight x values with solve_ivp RK45
    y_0 = np.concatenate((x_0, v_0), axis = None)
    init_sol = scipy.integrate.solve_ivp(charged_particle_eq, (t_0, t[7]), y_0, args = (B, U_grad), dense_output=True, method = "RK45", rtol=1e-10, atol=1e-12)
    x[:,:8] = init_sol.sol(t[0:8])[:3]
    v[:,:8] = init_sol.sol(t[0:8])[3:]

    

    for i in range(8, t.size ):


        # Compute F_n-1, F_n, F_n+1
        if i == 8:
            F_prev = F_j(A, x[:, (1): (6)],  h, U_grad)     # Need five x-values centered around 3
            F_n = F_j(A, x[:, (2): (7)], h, U_grad)         # Need five x-values centered around 4
            F_next = F_j(A, x[:, (3): (8)], h, U_grad)      # Need five x-values centered around 5

        if i > 8:
            F_prev = F_n                                        # Need five x-values centered around i-5
            F_n = F_next                                        # Need five x-values centered around i-4
            F_next = F_j(A, x[:, (i - 5): (i )], h, U_grad)     # Need five x-values centered around i-3

        F = (F_prev, F_n, F_next)      

        
        # Take a step
        x[:, i ] = fourth_order_sym_multistep_step(x[:, (i-8) : (i + 1)], alpha, h, beta0, beta1, F)

        print(x[:, i])
        # v is computed by a five point difference. Needs to be centered in i-2
        v[:, i - 2] = delta_yj(x[:, (i-4):(i+1)]) / h       
       

    return {"x": x, "v": v, "t": t}

