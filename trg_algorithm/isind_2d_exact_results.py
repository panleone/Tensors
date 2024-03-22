# This file contains a list of exact results for 2D Ising Model
# The boltzmann factor Kb is assumed to be 1
# Without loss of generality the coupling between spins J is also assumed to be 1

from math import log, sqrt, sinh, cosh, sqrt, cos, pi
import scipy.integrate as integrate

# Critical Temperature (J is assumed to be 1)
T_c =  2.0/log(1.0 + sqrt(2.0))
# Ising spins can take only 2 values + or - 1
n_spin_values = 2

# Exact magnetization
def exact_magnetization(temp):
    beta = 1.0 / temp
    if temp > T_c:
        return 0
    else:
        return (1.0 - sinh(2.0 * beta)**(-4.0))**0.125

# Exact free energy for the case in which there is no external field i.e h=0
# Solving the case h!=0 is an open problem
def exact_free_energy(temp):
    beta = 1.0 / temp
    cc, ss = cosh(2.0 * beta), sinh(2.0 * beta)
    k = 2.0 * ss / cc**2

    integral, err = integrate.quad(lambda x : 1.0 + sqrt(abs(1.0 - k * k * cos(x)**2)), 0, 0.5 * pi)
    result = integral / pi + log(cc) + 0.5 * log(2.0)
    return -result / beta