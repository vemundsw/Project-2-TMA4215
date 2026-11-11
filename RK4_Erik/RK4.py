import numpy as np

def RK4_step(t: np.array, Yn: np.array, system: np.array, h:float) -> np.array:
    """
    t: time (is ignored here)
    Yn: [x1,x2,x3,v1,v2,v3]
    system: gives dY/dt (RHS of differential equation)
    h: steplength

    returns: Yn_new

    """
    k1 = h * system(t, Yn)
    k2 = h * system(t, Yn + 0.5*k1)
    k3 = h * system(t, Yn + 0.5*k2)
    k4 = h * system(t, Yn + k3)

    Yn_new = Yn + 1/6*(k1 + 2*k2 + 2*k3 + k4)

    return Yn_new

def RK4(t: np.array, T: int,  Y0: np.array, system: np.array, h:float) -> np.array:
    """
    t: time (is ignored here)
    T: Time duration (how long it should run)
    Y0: initial Y vector
    system: RHS differential eqt
    h: step length
    """
    N = int(T/h)
    times = 0 + h * np.arange(N + 1)

    Y = np.zeros((6, N+1))
    Y[:, 0] = Y0

    for i in range(1, N+1):
        Y[:, i] = RK4_step(t, Y[:, i-1], system, h)
    
    return Y, times
