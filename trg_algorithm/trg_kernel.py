import numpy as np
from isind_2d_exact_results import n_spin_values
from utils import initial_trg_tensor, svd_decomposition

def trg_algorithm(temp, h, max_dimension, convergence_threshold, max_iterations = 30):
    # at each iteration t_tensor will have shape (current_dim, current_dim, current_dim, current_dim)
    current_dim = n_spin_values

    # current value of the partition function, updated at each iterative step
    current_z = 1
    # current value of the free energy
    current_f = np.log(current_z)
    # current number of particles in the Ising lattice
    current_particles = 1
    t_tensor = initial_trg_tensor(temp ,h)

    # Max number of iterations we are trying to do.
    # If the algorithm did not converge an error will be thrown
    for _ in range(max_iterations):

        # build the 4 f tensors with a svd decomposition
        f1, f3 = trg_svd_step(t_tensor.reshape([current_dim**2,current_dim**2]),
                              current_dim, max_dimension)
        f4, f2  = trg_svd_step(np.transpose(t_tensor, (0, 3, 2, 1)).reshape([current_dim**2,current_dim**2]),
                              current_dim, max_dimension)

        # Update the t_tensor by contracting the f tensors
        t_tensor = np.tensordot(
            np.tensordot(f2, f1, (1, 0)),
            np.tensordot(f3, f4, (1, 0)),
            ((1, 2), (1, 2)))
        t_tensor = np.transpose(t_tensor, (0,2,3,1))

        # Normalize by diving by its double trace, or the result would blow up iteration by iteration
        t_trace = np.einsum('ijij', t_tensor)
        t_tensor /= t_trace


        # Update thermodynamic values and check for convergence
        current_particles *= 2
        current_z *= pow(t_trace, 1./current_particles)
        new_f = np.log(current_z)
        if abs(new_f - current_f) < convergence_threshold:
            return new_f
        current_f = new_f
        # Update the dimension of the t_tensor
        current_dim = min(max_dimension, current_dim**2)
    # throw error as TRG did not converge
    raise Exception("TRG algorithm did not converge!")

# svd decomposition part performed in trg algorithm
def trg_svd_step(matrix, current_dim, max_dim):

    # perform svd decomposition
    u, s, v_dag = svd_decomposition(matrix, max_dim)

    # split the s tensor equally between u and v_dag
    u = u @ np.sqrt(s)
    v_dag = np.sqrt(s) @ v_dag

    # we are not interested in u and v but in their reshaped version
    last_dimension = u.size // current_dim**2
    f_u = u.reshape(current_dim, current_dim, last_dimension)
    f_v = v_dag.reshape(last_dimension, current_dim, current_dim)
    return f_u, f_v