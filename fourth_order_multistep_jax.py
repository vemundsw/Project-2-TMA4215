
import scipy
from scipy.differentiate import jacobian
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.lax as lax

jax.config.update("jax_enable_x64", True)

def charged_particle_eq(t: float, y: jnp.array, B: callable, U_grad: callable) -> jnp.array: 

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
    y_dot = jnp.cross(y[3:], B(y[:3])) - U_grad(y[:3])

    return jnp.concatenate((x1_dot, x2_dot, x3_dot, y_dot), axis = None)

# 2D charged particle 
x_0 = jnp.array([0, 1.0, 0.1])
v_0 = jnp.array([0.09, 0.05, 0.20])

def B_charged_particle(x: jnp.array) -> jnp.array:

    """
    Static magnetic field for 2D charged particles

    x: position on form (x1, x2, x3)

    Returns the magnetic field strength
    """

    return jnp.array([0, 0, jnp.sqrt(x[0] ** 2 + x[1] ** 2)])

@jax.jit
def A_charged_particle(x: jnp.array) -> jnp.array:

    """
    Vector potential for static magnetic field for 2D charged particle

    x: position on form (x1, x2, x3)
    
    Returns the vector potential
    """

    dist = jnp.sqrt(x[0] ** 2 + x[1] ** 2)

    return (1/3) * jnp.array([-x[1] * dist, x[0] * dist, 0 * dist])

@jax.jit
def U_charged_particle(x: jnp.array) -> float:
    """
    Electrostatic potential for 2D charged particle
    
    x: position on form (x1, x2, x3)
        
    Returns the potential
    """

    dist = jnp.sqrt(x[0] ** 2 + x[1] ** 2)

    return 1/ (100 * dist)

@jax.jit
def U_grad_charged_particle(x :jnp.array) -> jnp.array:
    """
    Gradient of electrostatic potential for 2D charged particle
        
    x: position on form (x1, x2, x3)
            
    Returns the gradient
    """

    dist = jnp.sqrt(x[0] ** 2 + x[1] ** 2) ** 3

    return -1/100 * jnp.array([x[0] / dist, x[1]/ dist, 0])


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

@jax.jit
def delta_yj(y: jnp.array) -> float:
    """
    Five point central difference centered in y[2]

    y: array on form (y_j-2, y_j-1, y_j, y_j+1, y_j+2). Each y has 3 components

    Returns the central difference
    """

    return 1/12 * (y[:,0] - 8 * y[:,1] + 8* y[:,3] - y[:,4]) 

@jax.jit
def F_j_jax(A_x :jnp.array, x :jnp.array,  h: float, U_grad_xj: jnp.array, A_jacobian: jnp.array) -> jnp.array:
    """
    Fourth order symmetric multistep method help function

    x: array with shape (x_j-2, x_j-1, x_j, x_j+1, x_j+2). Each x has 3 compontents
    h: steplength
    U_grad: potential gradient function

    Returns a 3 component array
    """

    x_j = x[:, 2]
    delta_Aj = delta_yj(A_x)/ h
    vj = delta_yj(x)/ h

    return  A_jacobian.T @ vj - delta_Aj - U_grad_xj

    
#fast_F_j = jax.jit(F_j_jax)


@jax.jit
def fourth_order_sym_multistep_step_fast(x: jnp.array, alpha: jnp.array, h: float, beta0: float, beta1: float, F: tuple) -> jnp.array:

    """
    Takes a step in the fourth order symmetrical multistep method

    x: array on form (x_n-4, x_n-3, ... , x_n+4)
    alpha: array on form (alpha_n-4, alpha_n-3, ... , alpha_n+4)
    h: steplength
    beta0, beta1: method parameters, 
    F: tuple on form (F_n-1, F_n, F_n+1)

    Returns an array x with updated x_n+4
    """

    prev_steps = jnp.sum(x * alpha[:-1], axis = 1)
    x_n4 = 1/ alpha[-1] *( h**2 * (beta1 * F[0] + beta0 * F[1] + beta1 * F[2]) - prev_steps)
    #x[-1] = x_n4

    return x_n4



def fourth_order_sym_multistep_lax(x: jnp.array, v: jnp.array, t_size: int, alpha: jnp.array, h: float, beta0: float, beta1: float, U_grad: callable, A: callable, jacobian: callable) -> jnp.array:

    """
    Fourth order symmetrical multistep method

    x: array with same shape as the final position array, except with the inital 8 values computed
    v: same as x but for velocity
    t_size: number of time points
    alpha: array on form (alpha_n-4, alpha_n-3, ... , alpha_n+4)
    h: steplength
    beta0, beta1: method parameters, 
    U_grad: potential gradient function
    A: magnetic field vector potential function
    Jacobian: Analytically computed jacobian of A 

    Returns a dict with x and v as keys
    """

    def scan_step(carry: tuple, _) -> jnp.array:

        """
        This function does the iterations in the multistep method. We initially implemented the 
        multistep method with numpy and a for loop here, but this took to much time. This function combined with lax.scan
        computes the values at every iteration simultanously

        carry: tuple of values which are updated for each iteration.
               Has the form (x_window, F_prev, F_n, F_next) 
               x_window: array of x_n-4 to x_n+3 (x_n+4 is the value we compute in each iteration)
               F_prev, F_n, F_next: F centered at x_n-1, x_n, x_n+1 respectively


        Returns new carry which are the updated iterative values, and x_last which is 
        an array of every x computed up to this point
        """

        x_window, v_current, F_prev, F_n, F_next = carry 

        # Compute new position
        x_last = fourth_order_sym_multistep_step_fast(x_window, alpha, h, beta0, beta1, (F_prev, F_n, F_next))

        # New window of x_n-4 to x_n+3 (x_n+4 is the value we compute in each iteration)
        new_x_window = jnp.concatenate([x_window[:, 1:], x_last[:, None]],axis=1)

        # Compute velocity as five point difference
        v_current =  delta_yj(new_x_window[:, -5:]) / h 

        # Update the F functions in each itearation
        F_prev = F_n                                        # Need five x-values centered around -5
        F_n = F_next                                        # Need five x-values centered around -4
        A_jacobi_next = jacobian(new_x_window[:, -3])
        U_grad_next = U_grad(new_x_window[:, -3])
        A_next = A(new_x_window[:, -5:])
        F_next = F_j_jax(A_next, new_x_window[:, -5:], h, U_grad_next, A_jacobi_next)     # Need five x-values centered around -3

        # Save current values for next iteration
        new_carry = (new_x_window, v_current, F_prev, F_n, F_next)
        
        return new_carry, (x_last, v_current)

    # Find initial values of the Fs
    A_jacobi_next = jacobian(x[:, 3])
    U_grad_next = U_grad(x[:, 3])
    A_next = A(x[:,  (1): (6)])
    F_prev = F_j_jax(A_next, x[:, (1): (6)],  h, U_grad_next, A_jacobi_next)     # Need five x-values centered around 3

    A_jacobi_next = jacobian(x[:, 4])
    U_grad_next = U_grad(x[:, 4])
    A_next = A(x[:,  (2): (7)])
    F_n = F_j_jax(A_next, x[:, (2): (7)], h, U_grad_next, A_jacobi_next)         # Need five x-values centered around 4

    A_jacobi_next = jacobian(x[:, 5])
    U_grad_next = U_grad(x[:, 5])
    A_next = A(x[:, (3): (8)])
    F_next = F_j_jax(A_next, x[:, (3): (8)], h, U_grad_next, A_jacobi_next)      # Need five x-values centered around 5

    
    # Inital values for lax.scan
    init_x_window = x[:, :8]
    initial_carry = (init_x_window, v[:, 7], F_prev, F_n, F_next)

    # lax.scan computes every value in the multistep iteration simultanously
    final_carry, final_xv = lax.scan(scan_step, initial_carry,  xs= None, length = t_size - 9)

    # Final x holds every x-value computed in the lax scan
    # x holds the 8 initial values
    x_full = jnp.concatenate([x[:, :8], final_xv[0].T],axis=1)
    v_full = jnp.concatenate([v[:, :8], final_xv[1].T],axis=1)

    return {"x": x_full, "v": v_full}

def create_grids(x_0: jnp.array, v_0: jnp.array,  h: float,  t_0: float, t_f: float, B: callable, U_grad: callable):
      
    # Create grids
    t = jnp.arange(t_0, t_f, h)
    x = jnp.zeros(( 3, t.size))
    v = jnp.zeros(( 3, t.size))
    
    # Find first eight x values with solve_ivp RK45
    y_0 = jnp.concatenate((x_0, v_0), axis = None)
    init_sol = scipy.integrate.solve_ivp(charged_particle_eq, (t_0, t[7]), y_0, args = (B, U_grad), dense_output=True, method = "RK45", rtol=1e-10, atol=1e-12)
    x = x.at[:,:8].set(jnp.asarray(init_sol.sol(t[0:8])[:3]))
    v = v.at[:,:8].set( jnp.asarray(init_sol.sol(t[0:8])[3:]))

    return x, v, t



#@jax.jit
# def fourth_order_sym_multistep_jax(x: jnp.array, v: jnp.array, t_size: int, alpha: jnp.array, h: float, beta0: float, beta1: float, t_0: float, t_f: float, U_grad: callable, A: callable, jacobian: callable) -> jnp.array:

#     """
#     Takes a step int the fourth order symmetrical multistep method

#     x: array on form (x_n-4, x_n-3, ... , x_n+4)
#     alpha: array on form (alpha_n-4, alpha_n-3, ... , alpha_n+4)
#     h: steplength
#     beta0, beta1: method parameters, 
#     F: array on form (F_n-1, F_n, F_n+1)
#     U_grad: potential gradient function
#     A: magnetic field vector potential function

#     Returns an array x with updated x_n+4
#     """

#     A_jacobi_next = jacobian(x[:, 3])
#     U_grad_next = U_grad(x[:, 3])
#     A_next = A(x[:,  (1): (6)])
#     F_prev = F_j_jax(A_next, x[:, (1): (6)],  h, U_grad_next, A_jacobi_next)     # Need five x-values centered around 3

#     A_jacobi_next = jacobian(x[:, 4])
#     U_grad_next = U_grad(x[:, 4])
#     A_next = A(x[:,  (2): (7)])
#     F_n = F_j_jax(A_next, x[:, (2): (7)], h, U_grad_next, A_jacobi_next)         # Need five x-values centered around 4

#     A_jacobi_next = jacobian(x[:, 5])
#     U_grad_next = U_grad(x[:, 5])
#     A_next = A(x[:, (3): (8)])
#     F_next = F_j_jax(A_next, x[:, (3): (8)], h, U_grad_next, A_jacobi_next)      # Need five x-values centered around 5


#     for i in range(8, t_size ):

#         # Compute F_n-1, F_n, F_n+1
#         F = (F_prev, F_n, F_next)      

        
#         # Take a step
#         x = x.at[:, i ].set( fourth_order_sym_multistep_step_fast(x[:, (i-8) : (i + 1)], alpha, h, beta0, beta1, F))

#         # v is computed by a five point difference. Needs to be centered in i-2
#         v = v.at[:, i - 2].set( delta_yj(x[:, (i-4):(i+1)]) / h   )    

#         F_prev = F_n                                        # Need five x-values centered around i-5
#         F_n = F_next                                        # Need five x-values centered around i-4
#         A_jacobi_next = jacobian(x[:, i + 1 -3])
#         U_grad_next = U_grad(x[:, i + 1 - 3])
#         A_next = A(x[:,  (i + 1- 5): (i + 1 )])
#         F_next = F_j_jax(A_next, x[:, (i + 1- 5): (i + 1 )], h, U_grad_next, A_jacobi_next)     # Need five x-values centered around i + 1 -3
       

#     return {"x": x, "v": v}

