# This file contains a list of utils functions needed to perform TRG
# The boltzmann factor Kb is assumed to be 1
# Without loss of generality the coupling between spins J is also assumed to be 1
from decimal import Decimal
from math import log, sqrt, sinh, cosh, sqrt, cos, pi
import numpy as np

# returns the initial tensor on which we apply the TRG algorithm
def initial_trg_tensor(temp, h):
    beta = 1.0 / temp

    a, b = sqrt(cosh(beta)), sqrt(sinh(beta))
    c = np.exp(0.25 * beta * h)

    w = np.array([[a * c, b * c],[a / c, -b / c]])
    return np.einsum("ia, ib, ic, id  -> abcd", w, w, w, w)

# performs the svd decomposition of the input matrix and returns the matrices U, S, V^{\dagger}
# it will also  truncate and keep only the rows and columns corresponding to the dim_max biggest singular values
def svd_decomposition(matrix, dim_max = Decimal('infinity')):
    u, s, v_dag = np.linalg.svd(matrix, full_matrices=True)
    s = np.diag(s)
    new_dimension = min(s.shape[0], dim_max)
    u = u[::, 0:new_dimension]
    v_dag = v_dag[0:new_dimension, ::]
    s = s[0:new_dimension, 0:new_dimension]
    return u, s, v_dag