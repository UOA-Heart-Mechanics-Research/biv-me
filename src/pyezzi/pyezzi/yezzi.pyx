import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport fabs

ctypedef np.float64_t DTYPE_FLOAT
ctypedef np.int_t DTYPE_INT
# ctypedef Py_ssize_t DTYPE_INT

# dirty fix to get rid of annoying warnings
np.seterr(divide='ignore', invalid='ignore')

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
def iterative_relaxation_3D(np.ndarray[DTYPE_INT, ndim=1] wall_idx_i,
                            np.ndarray[DTYPE_INT, ndim=1] wall_idx_j,
                            np.ndarray[DTYPE_INT, ndim=1] wall_idx_k,
                            np.ndarray[DTYPE_FLOAT, ndim=4] vectors,
                            DTYPE_FLOAT tolerance,
                            DTYPE_INT max_iterations,
                            np.ndarray[DTYPE_FLOAT, ndim=1] spacing,
                            np.ndarray[DTYPE_FLOAT, ndim=3] L0,
                            np.ndarray[DTYPE_FLOAT, ndim=3] L1,
                            tuple shape):
    cdef DTYPE_FLOAT hi, hj, hk
    hi, hj, hk = spacing

    cdef np.ndarray[DTYPE_FLOAT, ndim=4] abs_vectors = (
        np.abs(vectors).astype(np.float64))
    cdef np.ndarray[DTYPE_FLOAT, ndim=3] sum_abs_vectors = (
            abs_vectors[0] / hi
            + abs_vectors[1] / hj
            + abs_vectors[2] / hk).astype(np.float64)

    # Points of interest
    cdef DTYPE_INT n_points = len(wall_idx_i)
    cdef DTYPE_INT n_points2 = n_points * 2

    # cdef np.ndarray[DTYPE_FLOAT, ndim=3] L0 = np.zeros(shape, np.float64)
    # cdef np.ndarray[DTYPE_FLOAT, ndim=3] L1 = np.zeros(shape, np.float64)

    cdef np.ndarray[DTYPE_FLOAT, ndim=1] errors = np.zeros(n_points2,
                                                           np.float64)

    cdef bint convergence = False

    cdef DTYPE_FLOAT max_error = 1
    cdef DTYPE_INT iteration = 0

    cdef DTYPE_INT i = 0, j = 0, k = 0, n = 0, n2 = 0
    cdef DTYPE_FLOAT L0_i = 0, L0_j = 0, L0_k = 0, L1_i = 0, L1_j = 0, \
        L1_k = 0, prev_L0 = 0, prev_L1 = 0
    cdef DTYPE_FLOAT L0_value = 0, L1_value = 0, sum_abs_vector = 0

    while not convergence and iteration < max_iterations:
        iteration += 1
        for n in prange(n_points, nogil=True):
            i = wall_idx_i[n]
            j = wall_idx_j[n]
            k = wall_idx_k[n]
            if vectors[0, i, j, k] > 0:
                L0_i = abs_vectors[0, i, j, k] * L0[i - 1, j, k]
                L1_i = abs_vectors[0, i, j, k] * L1[i + 1, j, k]
            else:
                L0_i = abs_vectors[0, i, j, k] * L0[i + 1, j, k]
                L1_i = abs_vectors[0, i, j, k] * L1[i - 1, j, k]

            if vectors[1, i, j, k] > 0:
                L0_j = abs_vectors[1, i, j, k] * L0[i, j - 1, k]
                L1_j = abs_vectors[1, i, j, k] * L1[i, j + 1, k]
            else:
                L0_j = abs_vectors[1, i, j, k] * L0[i, j + 1, k]
                L1_j = abs_vectors[1, i, j, k] * L1[i, j - 1, k]

            if vectors[2, i, j, k] > 0:
                L0_k = abs_vectors[2, i, j, k] * L0[i, j, k - 1]
                L1_k = abs_vectors[2, i, j, k] * L1[i, j, k + 1]
            else:
                L0_k = abs_vectors[2, i, j, k] * L0[i, j, k + 1]
                L1_k = abs_vectors[2, i, j, k] * L1[i, j, k - 1]

            sum_abs_vector = sum_abs_vectors[i, j, k]
            prev_L0 = L0[i, j, k]
            prev_L1 = L1[i, j, k]
            L0_value = ((L0_i / hi
                         + L0_j / hj
                         + L0_k / hk
                         + 1)
                        / sum_abs_vector)
            L1_value = ((L1_i / hi
                         + L1_j / hj
                         + L1_k / hk
                         + 1)
                        / sum_abs_vector)
            if prev_L0 == 0:
                errors[n] = 1
            else:
                errors[n] = fabs((prev_L0 - L0_value) / prev_L0)
            n2 = n + n_points
            if prev_L1 == 0:
                errors[n2] = 1
            else:
                errors[n2] = fabs((prev_L1 - L1_value) / prev_L1)
            L0[i, j, k] = L0_value
            L1[i, j, k] = L1_value
        if iteration == 1:
            convergence = False
        else:
            convergence = has_converged(errors, n_points2, tolerance)

    return iteration, np.nanmax(errors)
