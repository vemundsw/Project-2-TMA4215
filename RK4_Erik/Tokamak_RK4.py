from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Differential_equations.Tokamak import system as system_Tokamak
from RK4_Erik.RK4 import RK4
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


