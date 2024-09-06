from stl import mesh as stl_mesh
import numpy as np
import pyvista as pv
import os
def export_to_stl(file_name,vertices, faces):
    # Create the mesh
    if '.stl' not in os.path.basename(file_name):
        ValueError(' filenma should include .stl extension')
    model_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            model_mesh.vectors[i][j] = vertices[int(f[j]), :]

    # Write the mesh to file
    model_mesh.save(file_name)

def export_points_to_cont6(filename,points,materials = None, scale = 1):
    # %% Write Nodes list in Continuity format
    print('Writing nodes form {0}'.format(filename))
    f = open(filename, 'w+')
    # Write headers
    node_string = 'Coords_1_val\tCoords_2_val\tCoords_3_val\tLabel\tNodes\n'
    # Write nodes
    for i,node in enumerate(points):
        for coord in node:
            node_string += '%f\t' % (coord*scale)
        node_string += '%i\t%i\n' % (materials[i], i + 1)
    f.write(node_string)
    f.close()

def export_elem_to_cont6(filename, elements, materials=None):
    # %% Write elements list in Continuity format
    print('Writing elements form {0}'.format(filename))
    f = open(filename+'.txt', 'w+')
    # Write headers
    elem_string = 'Node_0_Val\tNode_1_Val\tNode_2_Val\tLabel\tElement\n'


    for indx, elem in enumerate(elements):
        for node_id in elem:
            elem_string += '%i\t' % node_id
        elem_string += '%i\t%i\n' % (materials[indx], indx + 1)
    f.write(elem_string)
    f.close()

def export_model_to_cont6(self,filename, nodes, elements,
                          node_materials = None,
                          elem_materials = None, scale = 1):
    filename_nodes = filename + '_nodes'
    filename_elem = filename + '_elem'
    self.export_nodes_to_cont6(filename_nodes,nodes, node_materials, scale)
    self.export_elem_to_cont6(filename_elem, elements, elem_materials, scale)
    print('Continuity mesh exported')

def write_vtk_surface(filename: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Write a VTK surface mesh.

    Parameters
    ----------
    filename : The name of the output VTK file.
    vertices : An array of shape (N, 3) representing the vertex coordinates.
    faces : An array of shape (M, 3) representing the triangular faces.

    Returns
    -------
    None
    """

    if np.__version__ >= '1.20.0': # for compatibility with later versions of numpy
        np.bool = np.bool_

    mesh = pv.PolyData(vertices, np.c_[np.ones(len(faces)) * 3, faces].astype(int))
    mesh.save(filename, binary=False)

def write_colored_vtk_surface(filename: str, vertices: np.ndarray, faces: np.ndarray, colormat: np.ndarray) -> None:
    """
    Write a VTK surface mesh.

    Parameters
    ----------
    filename : The name of the output VTK file.
    vertices : An array of shape (N, 3) representing the vertex coordinates.
    faces : An array of shape (M, 3) representing the triangular faces.

    Returns
    -------
    None
    """

    if np.__version__ >= '1.20.0': # for compatibility with later versions of numpy
        np.bool = np.bool_

    mesh = pv.PolyData(vertices, np.c_[np.ones(len(faces)) * 3, faces].astype(int))
    mesh["colors"] = colormat
    mesh.save(filename, binary=False)

def export_to_obj(file_name: os.PathLike, vertices: np.ndarray, faces: np.ndarray) -> None:
    if '.obj' not in os.path.basename(file_name):
        ValueError(' filenma should include .obj extension')

    with open(file_name, 'w') as f:
        f.write("# OBJ file\n")
        for v in vertices:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        for p in faces:
            f.write("f")
            for i in p:
                f.write(" %d" % (i + 1))
            f.write("\n")