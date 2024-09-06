import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport fabs

ctypedef np.float64_t DTYPE_FLOAT
ctypedef np.int_t DTYPE_INT

# dirty fix to get rid of annoying warnings
np.seterr(divide='ignore', invalid='ignore')

# TODO: cimport this from pyezzi.laplace
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint has_converged(np.ndarray[DTYPE_FLOAT, ndim=1] errors,
                        DTYPE_INT n,
                        DTYPE_FLOAT tolerance):
    cdef bint res = True
    cdef DTYPE_INT i
    for i in prange(n, nogil=True):  # is prange useful here ?
        if errors[i] > tolerance:
            res = False
            break
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_3D(np.ndarray[DTYPE_INT, ndim=1] domain_idx_i,
             np.ndarray[DTYPE_INT, ndim=1] domain_idx_j,
             np.ndarray[DTYPE_INT, ndim=1] domain_idx_k,
             np.ndarray[DTYPE_FLOAT, ndim=3] init,
             DTYPE_FLOAT tolerance,
             DTYPE_INT max_iterations,
             np.ndarray[DTYPE_FLOAT, ndim=1] spacing):
    cdef np.ndarray[DTYPE_FLOAT, ndim=3] laplace_grid = np.copy(init)
    cdef DTYPE_INT iteration = 0
    cdef DTYPE_INT n_points = len(domain_idx_i)
    cdef DTYPE_INT i, j, k, n
    cdef DTYPE_FLOAT value
    cdef DTYPE_FLOAT hi, hj, hk, hi2, hj2, hk2, factor, prev_value
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] errors = np.zeros(n_points,
                                                           np.float64)
    cdef bint convergence = False

    hi, hj, hk = spacing
    hi2, hj2, hk2 = spacing ** 2

    factor = (hi2 * hj2 * hk2) / (2 * (hi2 * hj2 + hi2 * hk2 + hj2 * hk2))

    while not convergence and iteration < max_iterations:
        iteration += 1
        for n in prange(n_points, nogil=True):
            i = domain_idx_i[n]
            j = domain_idx_j[n]
            k = domain_idx_k[n]
            value = ((laplace_grid[i + 1, j, k] +
                      laplace_grid[i - 1, j, k]) / hi2 +
                     (laplace_grid[i, j + 1, k] +
                      laplace_grid[i, j - 1, k]) / hj2 +
                     (laplace_grid[i, j, k - 1] +
                      laplace_grid[i, j, k + 1]) / hk2) * factor
            prev_value = laplace_grid[i, j, k]
            laplace_grid[i, j, k] = value
            errors[n] = fabs((prev_value - value) / prev_value)

        if iteration == 1:
            convergence = False
        else:
            convergence = has_converged(errors, n, tolerance)

    return laplace_grid, iteration, np.nanmax(errors)
