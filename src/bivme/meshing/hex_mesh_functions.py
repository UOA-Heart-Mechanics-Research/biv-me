import numpy as np
from bivme.meshing import mesh
from copy import deepcopy
import warnings
from itertools import permutations

def extract_sudivided_hex_mesh(control_mesh, new_nodes_position, xi_coords,
                               node_elem_map):
    '''

    Args:
        control_mesh:
        new_nodes_position:
        xi_coords:
        node_elem_map:

    Returns:

    '''
    if isinstance(new_nodes_position, list):
        new_nodes_position = np.array(new_nodes_position)
    if isinstance(xi_coords, list):
        xi_position = np.array(xi_coords)
    sub_hex_mesh = deepcopy(control_mesh)
    sub_hex_mesh.label = 'hex_mesh' + control_mesh.label
    control_elem = control_mesh.elements
    # First we need to update the control mesh with the new positions
    # extracted from out data (xi_position and node_position)
    # select vertex new position
    vertex_index = np.prod(np.equal(xi_coords, 1) + np.equal(
        xi_coords, 0), axis=1).astype(bool)
    vertex_position = new_nodes_position[vertex_index]
    vertex_xi = xi_coords[vertex_index]
    vertex_elem_map = node_elem_map[vertex_index]
    updatd_hex_mesh = sub_hex_mesh.update_hex_node_position(
        vertex_position,
        vertex_xi,
        vertex_elem_map)

    # The nodes positions and xi coordinates are exported from BiV
    # fitting,
    # therefore the
    # mesh is subdivided only on xi0 and xi1. the xi2 axis can be
    # subdivided using linear interpolation.
    # We need to subdivide our control mesh twice to create the hex-mesh
    # topology

    sub_hex_mesh = updatd_hex_mesh.subdivide_linear_interpolation_hex(2)
    subdevided_elem = sub_hex_mesh.elements
    # After linear subdivision subdivision  the nodes position should be
    # updates with the given position
    # For that, we first need to create a map between the subdivided
    # mesh  nodes and their xi coordinates in control mesh
    # search edges at xi2 = 0 and xi2 = 1
    nb_control_elem = len(control_mesh.elements)
    np_control_nodes = len(control_mesh.nodes)
    local_elem_id = []
    xi0_elem = [0, 1, 8, 9, 9]
    xi01_elem = [[x + y for x in xi0_elem] for y in [0, 2, 16, 18, 18]]
    grid_elem = [[[x + z for x in k] for k in xi01_elem]
                 for z in [0, 4, 32, 36, 36]]
    grid_elem = np.array(grid_elem)
    grid_elem = grid_elem.swapaxes(0, 2)
    grid_local_vertex = np.zeros_like(grid_elem)
    grid_local_vertex[4, :, :] = 1
    grid_local_vertex[:, 4, :] = 2
    grid_local_vertex[:, :, 4] = 4

    grid_local_vertex[4, 4, :] = 3
    grid_local_vertex[:, 4, 4] = 6
    grid_local_vertex[4, :, 4] = 5

    grid_local_vertex[4, 4, 4] = 7

    xi_local = np.unique(xi_coords)
    nb_xi = xi_local.shape[0]
    elem_xi_cords = np.array([[xi0, xi1, xi2] for xi2 in xi_local
                              for xi1 in xi_local
                              for xi0 in xi_local])

    elem_xi_cords = elem_xi_cords.reshape((nb_xi, nb_xi, nb_xi, 3))
    elem_xi_cords = elem_xi_cords.swapaxes(0, 2)

    updated_node_position = deepcopy(sub_hex_mesh.nodes)

    for elem_id, elem in enumerate(control_elem):
        elem_index = np.where(np.equal(node_elem_map, elem_id))[0]
        elem_xi = xi_coords[elem_index]
        elem_verts = new_nodes_position[elem_index]
        elem_0 = elem_id * 64
        ind125 = subdevided_elem[elem_0 + grid_elem, grid_local_vertex]
        if elem_id == 148:
            print('stop')
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    xi = elem_xi_cords[i, j, k]
                    if np.equal(elem_xi, xi).all(axis=1).any():
                        new_position = elem_verts[
                            np.equal(elem_xi, xi).all(axis=1)]
                        node_index = ind125[i,j,k]

                        updated_node_position[node_index] = new_position[0]

    # Now all vertex on xi0 = [0,1] are updated
    # the inside node on xi0 = [0.25,0.5,0.75] should be linearly
    # interpolated  between  external faces
    sub_hex_mesh.nodes = updated_node_position
    for elem_id, elem in enumerate(control_elem):
        elem_0 = elem_id * 64
        ind125 = subdevided_elem[elem_0 + grid_elem, grid_local_vertex]
        for i in range(5):
            for j in range(5):
                u_elem_index = elem_0 + grid_elem[i, j, 4]
                l_elem_index = elem_0 + grid_elem[i, j, 0]
                u_local_node = grid_local_vertex[i, j, 4]
                l_local_node = grid_local_vertex[i, j, 0]
                upper_id = ind125[i,j,4]
                lower_id = ind125[i,j,0]

                for k in range(1, 4):
                    new_position = updated_node_position[lower_id] + \
                                   xi_local[k] * \
                                   (updated_node_position[upper_id] -
                                    updated_node_position[lower_id])

                    node_id = ind125[i,j,k]
                    updated_node_position[node_id] = new_position
    sub_hex_mesh.nodes = updated_node_position
    return sub_hex_mesh


# -*- coding: utf-8 -*-


def hermite_element_derivatives(hex_mesh):
    """
    HermiteElementDerivatives(mesh)

    Computes the nodal derivatives for tricubic Hermites by element from a twice
    subdivided hexahedral mesh.

    Input:
        mesh - where
            mesh.elements - (num_control_mesh_elements*64,8) Elements of a
            twice subdivided mesh as produced from:
                mesh = extract_sudivided_hex_mesh()
            mesh.nodes - (num_nodes,3) Nodes of the twice subdivided mesh

    Output bxe, bye, bze:
        (8,8,num_base_elements) - nodal derivitives for each element

        ['000', '000u', '000uw', '000w', '000t', '000wt', '000ut', '000uwt']
        ['100', '100u', '100uw', '100w', '100t', '100wt', '100ut', '100uwt']
        ['010', '010u', '010uw', '010w', '010t', '010wt', '010ut', '010uwt']
        ['110', '110u', '110uw', '110w', '110t', '110wt', '110ut', '110uwt']
        ['101', '101u', '101uw', '101w', '101t', '101wt', '101ut', '101uwt']
        ['101', '101u', '101uw', '101w', '101t', '101wt', '101ut', '101uwt']
        ['011', '011u', '011uw', '011w', '011t', '011wt', '011ut', '011uwt']
        ['111', '111u', '111uw', '111w', '111t', '111wt', '111ut', '111uwt']

    @author: Anna Mira
        adapted from hexblender:
        @author: Greg Sturgeon
        gregorymsturgeon@hotmail.com
    May 07, 2020
    """
    vertex = hex_mesh.nodes
    num_e = int(np.size(hex_mesh.elements, 0) / 64)
    elements= hex_mesh.elements
    # array of element node order such that
    local_elem_id = []
    xi0_elem = [0, 1, 8, 9, 9]
    xi01_elem = [[x + y for x in xi0_elem] for y in [0, 2, 16, 18, 18]]
    grid_elem = [[[x + z for x in k] for k in xi01_elem]
                 for z in [0, 4, 32, 36, 36]]
    grid_elem = np.array(grid_elem)
    grid_elem = grid_elem.swapaxes(0, 2)
    grid_local_vertex = np.zeros_like(grid_elem)
    grid_local_vertex[4, :, :] = 1
    grid_local_vertex[:, 4, :] = 2
    grid_local_vertex[:, :, 4] = 4

    grid_local_vertex[4, 4, :] = 3
    grid_local_vertex[:, 4, 4] = 6
    grid_local_vertex[4, :, 4] = 5

    grid_local_vertex[4, 4, 4] = 7


    # ind contains the indicies to the nodes in each element ordered:
    # ind[elementnumber,eta1,eta2,eta3]
    ind125 = np.zeros(( 5, 5, 5), int)
    uwt125 = np.zeros(( 5, 5, 5, 3), float)
    ind = np.zeros(( 4, 4, 4), int)
    uwt = np.zeros(( 4, 4, 4, 3), float)

    # c will contain a row of the 64 u,w,t products for each of the 64 points
    c = np.zeros((64, 64), float)
    xc = np.zeros((64), float)
    yc = np.zeros((64), float)
    zc = np.zeros((64), float)

    re = np.array([[0, 32, 8, 40, 2, 10, 34, 42],
                   [16, 48, 24, 56, 18, 26, 50, 58],
                   [4, 36, 12, 44, 6, 14, 38, 46],
                   [20, 52, 28, 60, 22, 30, 54, 62],

                   [1, 33, 9, 41, 3, 11, 35, 43],
                   [17, 49, 25, 57, 19, 27, 51, 59],
                   [5, 37, 13, 45, 7, 15, 39, 47],
                   [21, 53, 29, 61, 23, 31, 55, 63]])

    bxe = np.zeros((num_e, 8, 8), float)
    bye = np.zeros((num_e, 8, 8), float)
    bze = np.zeros((num_e, 8, 8), float)


    # create an array to map from the 4x4x4 array to a 64 element column vector
    i64 = np.array(list(range(0, 64)))
    i64 = i64.reshape(4, 4, 4)

    for k_e in range(num_e):
        ind125 = elements[k_e * 64 + grid_elem, grid_local_vertex]
        lu = np.sqrt(((vertex[ind125[ 0:4, :, :]] -
                       vertex[ind125[ 1:5, :, :]]) ** 2).sum(3))
        lw = np.sqrt(((vertex[ind125[ :, 0:4, :]] -
                       vertex[ind125[ :, 1:5, :]]) ** 2).sum(3))
        lt = np.sqrt(((vertex[ind125[ :, :, 0:4]] -
                       vertex[ind125[ :, :, 1:5]]) ** 2).sum(3))

        # Determine the lengths
        lu = np.cumsum(lu, axis=0)
        lw = np.cumsum(lw, axis=1)
        lt = np.cumsum(lt, axis=2)

        u = np.zeros((5, 5, 5, 1), float)
        w = np.zeros((5, 5, 5, 1), float)
        t = np.zeros((5, 5, 5, 1), float)

        for k in range(4):
            u[k + 1, :, :, 0] = lu[k, :, :] / lu[3, :, :]
            w[:, k + 1, :, 0] = lw[:, k, :] / lw[:, 3, :]
            t[:, :, k + 1, 0] = lt[:, :, k] / lt[:, :, 3]

        uwt125[ :, :, :, :] = np.concatenate((u, w, t), axis=3)

        # select 64 points and paramater values from the 125
        ind[ :, :, :] = ind125[
            np.ix_( [0, 1, 3, 4], [0, 1, 3, 4], [0, 1, 3, 4])].copy()
        uwt[:, :, :] = uwt125[
            np.ix_( [0, 1, 3, 4], [0, 1, 3, 4], [0, 1, 3, 4],
                      [0, 1, 2])].copy()
        x = vertex[ind[ :, :, :], 0]
        y = vertex[ind[ :, :, :], 1]
        z = vertex[ind[ :, :, :], 2]

        c = np.zeros((64, 64), float)
        # create the u,w,t products for each of the 64 points.
        for i1 in range(0, 4):
            for i2 in range(0, 4):
                for i3 in range(0, 4):
                    for j1 in range(1, 5):
                        for j2 in range(1, 5):
                            for j3 in range(1, 5):
                                c[i64[i1, i2, i3], i64[
                                    j1 - 1, j2 - 1, j3 - 1]] = \
                                    fH(j1, uwt[ i1, i2, i3, 0]) * \
                                    fH(j2, uwt[ i1, i2, i3, 1]) * \
                                    fH(j3, uwt[ i1, i2, i3, 2])
                                xc[i64[i1, i2, i3]] = x[i1, i2, i3]
                                yc[i64[i1, i2, i3]] = y[i1, i2, i3]
                                zc[i64[i1, i2, i3]] = z[i1, i2, i3]

        # Find the geometric coefficients bx,by,bz
        cinv = np.linalg.inv(c)
        bx = np.dot(cinv, xc)
        by = np.dot(cinv, yc)
        bz = np.dot(cinv, zc)

        # array to rearrange the 64 element column vectors into 8x8 array ordered as described in the comments


        for i in range(8):
            for j in range(8):
                bxe[k_e, i, j] = bx[re[i, j]]
                bye[k_e, i, j] = by[re[i, j]]
                bze[k_e, i, j] = bz[re[i, j]]


    return bxe, bye, bze


def fH(j, u):
    if j == 1:
        f = 2 * u ** 3 - 3 * u ** 2 + 1
    elif j == 2:
        f = -2 * u ** 3 + 3 * u ** 2
    elif j == 3:
        f = u ** 3 - 2 * u ** 2 + u
    elif j == 4:
        f = u ** 3 - u ** 2
    return f

def write_hermite_deriv(filename, Bx, By, Bz):
    #~ print ordered_hex_vertices[loop_num]
    # write headers
    deriv_string = 'u\tdu_dxi1\tdu_dxi2\td2u_dxi1xi2\tdu_dxi3\td2u_dxi2dxi3\td2u_dxi1dxi3\td3u_dxi\tCoord\tNode\tElement\n'
    # write elements
    for k in range(len(Bx)):
        for i in range(8):
            for j in range(8):
                deriv_string += '%f\t' % Bx[k,i,j]
            deriv_string += '%i\t%i\t%i\n' %(1,i+1,k+1)
            for j in range(8):
                deriv_string += '%f\t' % By[k,i,j]
            deriv_string += '%i\t%i\t%i\n' %(2,i+1,k+1)
            for j in range(8):
                deriv_string += '%f\t' % Bz[k,i,j]
            deriv_string += '%i\t%i\t%i\n' %(3,i+1,k+1)
    if filename is not None:
        myfile = open(filename+'.txt', 'w')
        myfile.write(deriv_string)
        myfile.close()
        print("Successfully exported %s" % filename)
        return None

def regularize_elements(mesh, iterations,
                            preserveRidges=True,
                            immobExtraord=False,
                            tangentMotion=True,
                            normalMotion=True,
                            internalMotion=True):
    # Working just for hex mesh using hex-blender
    linear_elem = deepcopy(mesh.elements)
    nodes = deepcopy(mesh.nodes)
    matlist = deepcopy(mesh.materials)
    new_elem_list = []
    for elem in linear_elem:
        new_elem_list.append([elem[0], elem[1], elem[3], elem[2], elem[4], elem[5], elem[7], elem[6]])
    new_elem_list = np.array(new_elem_list)
    new_nodes = regularize_elements(new_elem_list,nodes, iterations,
                                    preserveRidges,immobExtraord,
                                    tangentMotion, normalMotion,
                                    internalMotion )
    mesh.nodes = new_nodes

##def convert_linear_to_cubic_hermite_mesh(mesh):
##    # this function use hex-blender
##    linear_elem = deepcopy(mesh.elements)
##    nodes = deepcopy(mesh.nodes)
##    matlist = deepcopy(mesh.materials)
##    new_elem_list = []
##    for elem in linear_elem:
##        new_elem_list.append([elem[0], elem[1], elem[3], elem[2], elem[4], elem[5], elem[7], elem[6]])
##    new_matlist = matlist
##    for iteration in range(2):
##        subdivided_elem, subdivided_nodes = hex_interp_subdiv(np.array(new_elem_list),
##                                                          nodes, MatList=matlist,
##                                                          priorities=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
##                                                          thinPlateMapping =True)
##        old_matlist = new_matlist
##        new_matlist = []
##        for index, mat in enumerate(old_matlist):
##            new_matlist = new_matlist + [mat]*8
##        new_matlist = new_matlist
##    subdivided_elem, subdivided_nodes = hex_interp_subdiv(subdivided_elem,
##                                                          subdivided_nodes, MatList=matlist,
##                                                          priorities=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
##                                                          thinPlateRegions=[[1,2,3,4]],
##                                                          thinPlateMapping = True)
##    Bx, By, Bz = hermite_element_derivatives(subdivided_elem, subdivided_nodes)
##    Bx = np.swapaxes(Bx, 2, 0)
##    Bx = np.swapaxes(Bx, 2, 1)
##    By = np.swapaxes(By, 2, 0)
##    By = np.swapaxes(By, 2, 1)
##    Bz = np.swapaxes(Bz, 2, 0)
##    Bz = np.swapaxes(Bz, 2, 1)
##    for i in range(len(Bx)):
##        tmp = Bx[i, :, 3].copy()
##        Bx[i, :, 3] = Bx[i, :, 4].copy()
##        Bx[i, :, 4] = tmp.copy()
##        tmp = Bx[i, :, 5].copy()
##        Bx[i, :, 5] = Bx[i, :, 6].copy()
##        Bx[i, :, 6] = tmp.copy()
##        tmp = Bx[i, 2, :].copy()
##        Bx[i, 2, :] = Bx[i, 3, :].copy()
##        Bx[i, 3, :] = tmp.copy()
##        tmp = Bx[i, 6, :].copy()
##        Bx[i, 6, :] = Bx[i, 7, :].copy()
##        Bx[i, 7, :] = tmp.copy()
##    for i in range(len(By)):
##        tmp = By[i, :, 3].copy()
##        By[i, :, 3] = By[i, :, 4].copy()
##        By[i, :, 4] = tmp.copy()
##        tmp = By[i, :, 5].copy()
##        By[i, :, 5] = By[i, :, 6].copy()
##        By[i, :, 6] = tmp.copy()
##        tmp = By[i, 2, :].copy()
##        By[i, 2, :] = By[i, 3, :].copy()
##        By[i, 3, :] = tmp.copy()
##        tmp = By[i, 6, :].copy()
##        By[i, 6, :] = By[i, 7, :].copy()
##        By[i, 7, :] = tmp.copy()
##    for i in range(len(Bz)):
##        tmp = Bz[i, :, 3].copy()
##        Bz[i, :, 3] = Bz[i, :, 4].copy()
##        Bz[i, :, 4] = tmp.copy()
##        tmp = Bz[i, :, 5].copy()
##        Bz[i, :, 5] = Bz[i, :, 6].copy()
##        Bz[i, :, 6] = tmp.copy()
##        tmp = Bz[i, 2, :].copy()
##        Bz[i, 2, :] = Bz[i, 3, :].copy()
##        Bz[i, 3, :] = tmp.copy()
##        tmp = Bz[i, 6, :].copy()
##        Bz[i, 6, :] = Bz[i, 7, :].copy()
##        Bz[i, 7, :] = tmp.copy()
##    new_elem_list = []
##    for elem in subdivided_elem:
##        new_elem_list.append([elem[0], elem[1], elem[3], elem[2], elem[4], elem[5], elem[7], elem[6]])
##    sub_div_mesh = Mesh('sub_dic_mesh')
##    # new_elem = np.vstack((biv_model.elements,subdivided_elem),axis= 0)
##    sub_div_mesh.set_elements(new_elem_list)
##    sub_div_mesh.set_nodes(subdivided_nodes)
##    sub_div_mesh.set_materials(range(len(new_elem_list)),new_matlist)
##    return sub_div_mesh, Bx, By, Bz