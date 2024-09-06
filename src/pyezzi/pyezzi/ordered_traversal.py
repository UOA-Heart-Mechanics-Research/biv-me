from multiprocessing import Pool
import logging

log = logging.getLogger(__name__)

import numpy as np

from scipy.ndimage.morphology import binary_dilation
from heapq import heappush, heappop

UNVISITED = 0
VISITED = 1
SOLVED = 2

CUBE = np.ones((3, 3, 3), bool)


def ordered_traversal(gradients,
                      wall,
                      epi,
                      endo,
                      wall_i,
                      wall_j,
                      wall_k,
                      spacing):
    abs_vectors = np.abs(gradients)
    sum_abs_vectors = (abs_vectors[0] / spacing[0]
                       + abs_vectors[1] / spacing[1]
                       + abs_vectors[2] / spacing[2])

    log.debug(f"Ordered traversal")
    inner_mask = (binary_dilation(endo, CUBE) ^ endo) & wall
    outer_mask = binary_dilation(~epi) & epi

    with Pool() as p:
        L0, L1 = p.starmap(
            Lx,
            [(inner_mask,
              gradients,
              abs_vectors,
              sum_abs_vectors,
              wall,
              wall_i,
              wall_j,
              wall_k,
              spacing),
             (outer_mask,
              -gradients,
              abs_vectors,
              sum_abs_vectors,
              wall,
              wall_i,
              wall_j,
              wall_k,
              spacing)]
        )

    return L0, L1


def Lx(border,
       gradients,
       abs_vectors,
       sum_abs_vectors,
       wall,
       wall_i,
       wall_j,
       wall_k,
       spacing):
    max_i, max_j, max_k = np.array(wall.shape) - 1
    hi, hj, hk = spacing

    flat_table = np.zeros_like(wall, np.int)
    flat_table[wall] = np.arange(len(wall_i))

    status_flat = np.zeros(len(wall_i), np.uint8)
    flat_Lx = np.zeros(len(wall_i))

    Lx = np.zeros(wall.shape)

    for a, sp in enumerate(spacing):
        for sh in 1, -1:
            Lx[np.roll(wall, shift=sh, axis=a)] -= sp
    Lx /= 2
    Lx.clip(min=-np.mean(spacing) * 0.5, out=Lx)

    status = np.zeros_like(wall, np.uint8)

    heap = []

    log.debug(f"{border.sum()} points on the border...")
    for m, n in enumerate(flat_table[border]):
        i, j, k = (wall_i[n],
                   wall_j[n],
                   wall_k[n])
        if gradients[0, i, j, k] > 0:
            Lx_i = abs_vectors[0, i, j, k] * Lx[i - 1, j, k]
        else:
            Lx_i = abs_vectors[0, i, j, k] * Lx[i + 1, j, k]

        if gradients[1, i, j, k] > 0:
            Lx_j = abs_vectors[1, i, j, k] * Lx[i, j - 1, k]
        else:
            Lx_j = abs_vectors[1, i, j, k] * Lx[i, j + 1, k]

        if gradients[2, i, j, k] > 0:
            Lx_k = abs_vectors[2, i, j, k] * Lx[i, j, k - 1]
        else:
            Lx_k = abs_vectors[2, i, j, k] * Lx[i, j, k + 1]

        Lx_value = (Lx_i / hi
                    + Lx_j / hj
                    + Lx_k / hk + 1) / sum_abs_vectors[i, j, k]

        flat_Lx[n] = Lx_value
        Lx[i, j, k] = Lx_value
        status_flat[n] = VISITED
        status[i, j, k] = VISITED
        heappush(heap, (Lx_value, n))

    log.debug("Other points...")
    to_solve = len(status_flat)
    heapempty = False
    for iteration in range(to_solve):
        if iteration % 100000 == 0:
            log.debug(f"UNVISITED {(status_flat == UNVISITED).sum()}")
            log.debug(f"VISITED   {(status_flat == VISITED).sum()}")
            log.debug(f"SOLVED    {(status_flat == SOLVED).sum()}")
        while True:
            try:
                n = heappop(heap)[1]
            except IndexError:
                heapempty = True
                break
            if status_flat[n] == VISITED:
                break
        if heapempty:
            log.warning(f"Stopped at iteration {iteration + 1}/{to_solve} "
                        f"because the heap was empty")
            log.debug(f"UNVISITED {(status_flat == UNVISITED).sum()}")
            log.debug(f"VISITED   {(status_flat == VISITED).sum()}")
            log.debug(f"SOLVED    {(status_flat == SOLVED).sum()}")
            break
        i0, j0, k0 = wall_i[n], wall_j[n], wall_k[n]
        status_flat[n] = SOLVED
        status[i0, j0, k0] = SOLVED
        for i in i0 - 1, i0, i0 + 1:
            for j in j0 - 1, j0, j0 + 1:
                for k in k0 - 1, k0, k0 + 1:
                    if i < 0 or j < 0 or k < 0:
                        continue
                    if i > max_i or j > max_j or k > max_k:
                        continue
                    if not wall[i, j, k]:
                        continue
                    if gradients[0, i, j, k] > 0:
                        Lx_i = abs_vectors[0, i, j, k] * Lx[i - 1, j, k]
                    else:
                        Lx_i = abs_vectors[0, i, j, k] * Lx[i + 1, j, k]

                    if gradients[1, i, j, k] > 0:
                        Lx_j = abs_vectors[1, i, j, k] * Lx[i, j - 1, k]
                    else:
                        Lx_j = abs_vectors[1, i, j, k] * Lx[i, j + 1, k]

                    if gradients[2, i, j, k] > 0:
                        Lx_k = abs_vectors[2, i, j, k] * Lx[i, j, k - 1]
                    else:
                        Lx_k = abs_vectors[2, i, j, k] * Lx[i, j, k + 1]

                    Lx_value = (Lx_i / hi
                                + Lx_j / hj
                                + Lx_k / hk + 1) / sum_abs_vectors[i, j, k]
                    n = flat_table[i, j, k]
                    Lx[i, j, k] = Lx_value
                    flat_Lx[n] = Lx_value
                    if status[i, j, k] == UNVISITED:
                        status[i, j, k] = VISITED
                        status_flat[n] = VISITED
                    heappush(heap, (Lx_value, n))

    Lx[~wall] = 0

    return Lx
