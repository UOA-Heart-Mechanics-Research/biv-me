from stl import mesh as stl_mesh
import numpy as np

import os
def export_to_stl(file_name,vertices, faces):
    # Create the mesh
    if '.stl' not in os.path.basename(file_name):
        ValueError(' filenma should include .stl extension')
    model_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            model_mesh.vectors[i][j] = vertices[int(f[j]), :]

    # Write the mesh to file "cube.stl"
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

