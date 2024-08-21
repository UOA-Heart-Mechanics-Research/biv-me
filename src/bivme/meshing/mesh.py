import os
import numpy as np
import copy
import _pickle as pickle
from copy import deepcopy
from .geometric_tools import normalize_v3
from scipy.spatial import cKDTree


class Mesh:
    """
        Attributes:
          nb_nodes                      Number of control nodes.
          nb_elements                   Number of elements.
          label                         mesh label

          nodes                         Array of x,y,z coordinates of nodes).
          elements                      Elements connectivities
    """
    def __init__(self,mesh_label, node_file_name = None, element_file_name = None, nodes_basis = 4):

        """

        Args:
            node_file_name: str giving the file name where to read mesh node
            element_file_name: str giving the file name where to read element connectivity
        """

        """
                        6 ----------- 7
                        /|           /|                   
                       4 ----------- 5|
                       | |          | |
                       | |          | |
                       | 2 ---------|-3
                       |/           |/
                       0 -----------1        
        elem = [0,1,2,3,4,5,6,7]               
                     
        """
        self.label = mesh_label
        self.nodes_basis = nodes_basis
        if (node_file_name == None) :
            self.nodes=np.empty((0,0))
            self.nb_nodes = 0

        elif (not os.path.exists(node_file_name)):
            ValueError('Node file does not exist')
        else:
            first_line = np.genfromtxt(node_file_name,max_rows=1,
                                       usecols=(0,1,2))
            skip_header = (1 if np.isnan(first_line).any() else  0)

            self.nodes = np.genfromtxt(node_file_name,skip_header=skip_header,
                                       usecols=(0,1,2)).astype(float)
            self.nb_nodes = self.nodes.shape[0]

        if element_file_name is None:
            self.elements = np.empty((0, 0))
            self.nb_elements = 0
            self.materials = np.empty((0,0))

        elif os.path.exists(element_file_name):
            first_line = np.genfromtxt(element_file_name,max_rows=1,
                                       usecols=(0,1,2))
            skip_header = (1 if np.isnan(first_line).any() else 0)

            self.elements = np.genfromtxt(element_file_name,skip_header=skip_header,
                                          usecols=range(nodes_basis)).astype(int)-1
            self.nb_elements = self.elements.shape[0]
            try:
                self.materials = np.genfromtxt(element_file_name, skip_header=skip_header,
                                              usecols=nodes_basis).astype(int) - 1
            except:
                self.materials = np.zeros(self.elements.shape[0])

    def swap_elem_nodes(self,input_index, output_index):
        new_elem = deepcopy(self.elements)
        for elem_index,elem in enumerate(self.elements):
            new_elem[elem_index,output_index] = elem[input_index]
            new_elem[elem_index,input_index] = elem[output_index]
        self.elements = new_elem

    def set_elements(self, new_elem_array):

        if isinstance(new_elem_array,list):
            self.elements = copy.deepcopy( np.array(new_elem_array))
        else:
            self.elements = copy.deepcopy( new_elem_array)
        self.nb_elements = self.elements.shape[0]
        self.materials = np.zeros(self.elements.shape[0])

    def set_nodes(self, new_nodes_array):
        if isinstance(new_nodes_array,list):
            self.nodes =copy.deepcopy( np.array(new_nodes_array))
        else:
            self.nodes = new_nodes_array
        self.nb_nodes = copy.deepcopy(self.nodes.shape[0])
    def set_materials(self,elem_index,matlist):

        if not (len(elem_index) == len(matlist)):
            ValueError('list of elements and materials should be of same leght')
            return
        if len(self.materials) != len(self.elements):
            self.materials = np.zeros_like(self.elements)
        for index,elem in enumerate(elem_index):
            self.materials[int(elem)] = matlist[index]

    def get_materials(self):
        return copy.deepcopy(self.materials)

    def get_mesh_component(self, materials, label = None, reindex_nodes = True):
        if label is None:
            new_component = Mesh('sub_component')
        else:
            new_component = Mesh(label)

        if np.isscalar(materials):
            materials = [materials]
        if not isinstance(materials,list):
            materials = list(materials)
        elem =[]
        new_materials = []

        for m in materials:
            elem.append(self.elements[self.materials == m].astype(int))
            new_materials = new_materials + [m]* np.sum(self.materials == m)
        elem = np.array([x for sublist in elem for x in sublist])
        new_materials = np.array(new_materials)
        nodes_index = np.unique(elem)
        nodes = self.nodes
        if reindex_nodes:
            nodes = self.nodes[nodes_index]
            for index,n in enumerate(nodes_index):
                elem[elem == n] = index

        new_component.set_nodes(nodes)
        new_component.set_elements(elem)
        new_component.set_materials(list(range(len(elem))),new_materials)
        return new_component

    def get_surface(self, elem_list=[], xi = 'all', output_elem_type ='triangle'):
        """
        Extract mesh surface from a 3D hex surface giving the xi position :
        Args:
            elem_list: subgroup of elements to extract the surface from
            xi: element face to be extracted (xi1, xi2, xi3 or all)
            output_elem_type: type of output face ('triangle' or 'quadrilateral')

        Returns:

        """
        if len(elem_list) ==0:
            elements_array = self.elements
        else:
            elements_array = self.elements[elem_list]

        if isinstance(elements_array, list):
            elements_array_local = np.array(elements_array)
        else:
            elements_array_local = elements_array


        if output_elem_type == 'triangle':
            triangular_surface = np.empty((0, 3))
            for element in elements_array_local:
                if element.shape[0] == 3:
                    triangular_surface = np.vstack((triangular_surface,element))

                if element.shape[0] == 6:  # tetrahedron type
                    print('Not implemented')
                if element.shape[0] == 8:

                    if xi == 'xi1' or xi == 'all':
                        triangular_surface = np.vstack((triangular_surface,
                            element[:3]))
                        triangular_surface = np.vstack((triangular_surface,
                            element[1:4]))
                        triangular_surface = np.vstack((triangular_surface,
                            element[-3:]))
                        triangular_surface = np.vstack((triangular_surface,
                            element[-4:-1]))
                    if xi =='xi2' or xi == 'all':
                        triangular_surface = np.vstack((triangular_surface,
                            element[[0, 1, 4]]))
                        triangular_surface = np.vstack((triangular_surface,
                            element[[1, 4, 5]]))
                        triangular_surface = np.vstack((triangular_surface,
                            element[[2, 3, 6]]))
                        triangular_surface = np.vstack((triangular_surface,
                            element[[3, 6, 7]]))
                    if xi =='xi3' or xi == 'all':
                        triangular_surface = np.vstack((triangular_surface,
                           element[[0, 2, 4]]))
                        triangular_surface = np.vstack((triangular_surface,
                           element[[2, 4, 6]]))
                        triangular_surface = np.vstack((triangular_surface,
                           element[[1, 3, 5]]))
                        triangular_surface = np.vstack((triangular_surface,
                            element[[3, 5, 7]]))
        else:
            triangular_surface = np.empty((0, 4))
            for element in elements_array_local:
                if element.shape[0] == 8:
                    if xi == 'xi1' or xi == 'all':
                        triangular_surface = np.vstack((triangular_surface, element[:4]))
                        triangular_surface = np.vstack(
                            (triangular_surface, element[-4:]))
                    if xi == 'xi2' or xi == 'all':
                        triangular_surface = np.vstack(
                            (triangular_surface, element[[0, 1, 4, 5]]))

                        triangular_surface = np.vstack(
                            (triangular_surface, element[[2, 3, 6, 7]]))

                    if xi == 'xi3' or xi == 'all':
                        triangular_surface = np.vstack(
                            (triangular_surface, element[[0, 2, 4, 6]]))

                        triangular_surface = np.vstack(
                            (triangular_surface, element[[1, 3, 5, 7]]))
                else:
                    ValueError('Invalid element type')
                    return

        return triangular_surface


    def get_lines(self):
        '''
        Return array of mesh edges as shape nx2
        :return:
        '''
        elements_array = self.elements
        if isinstance(elements_array, list):
            elements_array_local = np.array(elements_array)
        else:
            elements_array_local = elements_array

        lines_list = []
        for element in elements_array_local:
            if element.shape[0] == 3:  # triangular mesh
                lines_list.append(tuple(element[[0, 1]]))
                lines_list.append(tuple(element[[1, 2]]))
                lines_list.append(tuple(element[[2, 0]]))

            if element.shape[0] == 6:  # triangular prism type
                print('Not implemented')
            if element.shape[0] == 8:  # linear Hexahedron
                lines_list.append(tuple(element[[0, 1]]))
                lines_list.append(tuple(element[[0, 2]]))
                lines_list.append(tuple(element[[0, 4]]))
                lines_list.append(tuple(element[[3, 1]]))
                lines_list.append(tuple(element[[3, 2]]))
                lines_list.append(tuple(element[[3, 7]]))
                lines_list.append(tuple(element[[5, 4]]))
                lines_list.append(tuple(element[[5, 7]]))
                lines_list.append(tuple(element[[5, 1]]))
                lines_list.append(tuple(element[[6, 4]]))
                lines_list.append(tuple(element[[6, 7]]))
                lines_list.append(tuple(element[[6, 2]]))

        return np.array([list(x) for x in set(lines_list)])

    def get_nodes(self):
        return self.nodes

    def export_pickle(self,file_name):
        mpickle = dict()
        mpickle['verts'] = self.nodes.tolist()
        mpickle['elems'] = self.elements.tolist()
        mpickle['mats'] = self.get_materials().tolist()
        pickle_string = pickle.dumps(mpickle)

        fid_pickle = open(file_name + '.pickle', 'wb')
        fid_pickle.write(pickle_string)
        fid_pickle.close()



    def subdivide_linear_interpolation_hex(self,iterations):
        '''
        Perform interpolative subdivision for a 3D hex mesh
        :param iterations: int given the numer of iterations
        :return: new_mesh of type mesh
        '''
        new_mesh = deepcopy(self)
        if len(self.elements[0]) !=8:
            print('Only hex elements is implemented')
            return
        edge_def = np.array(
            [[0, 1],[0, 2], [2, 3], [1, 3], [4, 5], [4, 6], [6, 7],
             [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]])
        face_def = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5], [2, 3, 6, 7],
             [0, 2, 4, 6], [1, 3, 5, 7]])

        for iter in range(iterations):
            old_mesh =deepcopy(new_mesh)
            num_verts = old_mesh.nodes.shape[0]
            num_elem = old_mesh.elements.shape[0]
            verts= old_mesh.nodes
            elements = old_mesh.elements
            mats =old_mesh.materials
            edge_point = {}
            face_point = {}
            elem_point = np.zeros((num_elem, 3))
            for k,elem in enumerate(elements):
                for i in range(12):
                    edge_ind = (min(elem[edge_def[i][0]],elem[edge_def[i][1]]),
                            max(elem[edge_def[i][0]],elem[edge_def[i][1]]))
                    edge_point[edge_ind] = .5 * (verts[elem[edge_def[i][0]]]
                                            + verts[elem[edge_def[i][1]]])
                for i in range(6):
                    face = [elem[face_def[i][0]], elem[face_def[i][1]],
                            elem[face_def[i][2]], elem[face_def[i][3]]]
                    face.sort()
                    face_ind = tuple(face)
                    face_point[face_ind] = .25 * (verts[face[0]] + verts[face[ 1]]
                                                  + verts[face[2]] + verts[face[3]])
                elem_point[k] = (1.0 / 8.0) * verts[elem].sum(0)


            edge_point_id= {}
            edge_id = list(edge_point.keys())
            num_edges = len(edge_id)
            for k,edge in enumerate(edge_id):
                edge_point_id[edge] = num_verts+k
            edge_point = np.array(list(edge_point.values()))

            face_id = list(face_point.keys())
            num_faces = len(face_id)
            face_point_id ={}
            for k,face in enumerate(face_id):
                face_point_id[face] = num_verts + num_edges + k
            elem_point_id =[]
            for k in range(elem_point.shape[0]):
                elem_point_id.append( num_verts + num_edges +num_faces +k)
            face_point = np.array(list(face_point.values()))
            new_vertex = np.concatenate((verts,edge_point,face_point,
                                         elem_point),axis=0)


            for k,elem in enumerate(elements):

                elem_edge = [
                        (min(elem[edge[0]], elem[edge[1]]),
                        max(elem[edge[0]], elem[edge[1]])) for edge in edge_def]

                elem_face =[[elem[face[0]], elem[face[1]], elem[face[2]],
                            elem[face[3]]] for face in face_def]

                elem_face = [tuple(np.sort(face)) for face in elem_face]

                eN1 = np.array(
                    [elem [0], edge_point_id[elem_edge[0]],
                     edge_point_id[elem_edge[1]], face_point_id[elem_face[0]],
                     edge_point_id[elem_edge[8]],  face_point_id[elem_face[2]],
                     face_point_id[elem_face[4]], elem_point_id[k]])
                eN2 = np.array(
                    [edge_point_id[elem_edge[0]], elem[1],
                     face_point_id[elem_face[0]], edge_point_id[elem_edge[3]],
                     face_point_id[elem_face[2]], edge_point_id[elem_edge[9]],
                     elem_point_id[k], face_point_id[elem_face[5]]])
                eN3 = np.array(
                    [edge_point_id[elem_edge[1]], face_point_id[elem_face[0]],
                     elem[2],edge_point_id[elem_edge[2]],
                     face_point_id[elem_face[4]], elem_point_id[k],
                     edge_point_id[elem_edge[10]],face_point_id[elem_face[3]]])
                eN4 = np.array(
                    [face_point_id[elem_face[0]], edge_point_id[elem_edge[3]],
                     edge_point_id[elem_edge[2]], elem[3],
                     elem_point_id[k], face_point_id[elem_face[5]],
                     face_point_id[elem_face[3]],edge_point_id[elem_edge[11]]])
                eN5 = np.array(
                    [edge_point_id[elem_edge[8]], face_point_id[elem_face[2]],
                     face_point_id[elem_face[4]], elem_point_id[k],
                     elem[4],edge_point_id[elem_edge[4]],
                     edge_point_id[elem_edge[5]], face_point_id[elem_face[1]]])
                eN6 = np.array(
                    [face_point_id[elem_face[2]], edge_point_id[elem_edge[9]],
                     elem_point_id[k], face_point_id[elem_face[5]],
                     edge_point_id[elem_edge[4]],elem[5],
                     face_point_id[elem_face[1]], edge_point_id[elem_edge[7]]])
                eN7 = np.array(
                    [face_point_id[elem_face[4]],elem_point_id[k],
                     edge_point_id[elem_edge[10]],face_point_id[elem_face[3]],
                     edge_point_id[elem_edge[5]], face_point_id[elem_face[1]],
                     elem[6],edge_point_id[elem_edge[6]]])
                eN8 = np.array(
                    [ elem_point_id[k], face_point_id[elem_face[5]],
                    face_point_id[elem_face[3]],edge_point_id[elem_edge[11]],
                    face_point_id[elem_face[1]],edge_point_id[elem_edge[7]],
                      edge_point_id[elem_edge[6]],elem[7]])
                eN1.resize(1, 8)
                eN2.resize(1, 8)
                eN3.resize(1, 8)
                eN4.resize(1, 8)
                eN5.resize(1, 8)
                eN6.resize(1, 8)
                eN7.resize(1, 8)
                eN8.resize(1, 8)
                if k == 0:
                    new_elem = np.concatenate(
                        (eN1, eN2, eN3, eN4, eN5, eN6, eN7, eN8), axis=0)
                    if len(mats) == len(elements):
                        new_mats =  [mats[k]]*8
                else:
                    new_elem = np.concatenate(
                        (new_elem, eN1, eN2, eN3, eN4, eN5, eN6, eN7, eN8), axis=0)
                    if len(mats) == len(elements):
                        new_mats = new_mats + [mats[k]]*8
            new_mesh = Mesh('subdivided1')
            new_mesh.elements = new_elem
            new_mesh.nodes = new_vertex
            new_mesh.set_materials(list(range(len(new_elem))), new_mats)
        return new_mesh

    def update_hex_node_position(self,new_node_position, xi_position, \
                                                    node_elem_map):
        """

        Args:
            new_nodes_position:  nX3 array with node position
            hex_mesh: mesh object, containing elements and nodes position of a
                quad/hex mesh (for subdivision model, the hex_mesh corresponds
                to the initial control mesh )
            xi_position: nx3 array, xi_position of each new node (as
                        exported from BiVFitting mesh)
            elem_points: n array, element number in the quad mesh for each node of
                the tetrahedral mesh

        Returns:
            new_hex_mesh: mesh object with updated nodes position

        """
        new_mesh = deepcopy(self)
        new_mesh.label ='updated_'+self.label
        xi_order = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        for elem_id, elem_nodes in enumerate(self.elements):

            elem_index = np.equal(node_elem_map, elem_id)
            enode_position = new_node_position[elem_index]
            enode_xi = xi_position[elem_index]


            for node_index, node in enumerate(elem_nodes):

                index = np.prod(enode_xi == xi_order[node_index],
                                axis=1).astype(bool)
                if index.any():
                    new_mesh.nodes[node] = enode_position[index][0]
        return new_mesh

    def export_nodes_cont6(self, filename, scale = 1):
        # %% Write Nodes list in cont6 format
        print('Writing nodes form {0}'.format(filename))

        f = open(filename+'.txt', 'w+')
        # Write headers
        node_string = 'Coords_1_val\tCoords_2_val\tCoords_3_val\tLabel\tNodes\n'

        # Write nodes
        for i,node in enumerate(self.nodes):
            for coord in node:
                node_string += '%f\t' % (coord*scale)
            node_string += '%i\t%i\n' % (i + 1, i + 1)

        f.write(node_string)
        f.close()

    def export_elements_cont6(self, filename , elem_type = '3D'):
        """
        Write mesh in continuity format
        Args:
            filename: output file name
            elem_type: export element type : '2D' for quadrilateral or
                '3D' for hexahedron)

        Returns:
        """

        # %% Write Nodes list in Continuity format
        print('Writing elements form {0}'.format(filename))
        nb_of_nodes = self.elements.shape[1]
        if not (nb_of_nodes ==4 or nb_of_nodes ==8):
            print('Only mesh with quadrilateral elements or hexhedral '
                  'elements are implemented')
            return
        f = open(filename+'.txt', 'w+')
        # Write headers
        if elem_type == '3D':
            elem_string = 'Node_0_Val\tNode_1_Val\tNode_2_Val\tNode_3_Val\t' \
                      'Node_4_Val\tNode_5_Val\tNode_6_Val\tNode_7_Val\tLabel\tElement\n'
        elif elem_type == '2D':
            elem_string = 'Node_0_Val\tNode_1_Val\tNode_2_Val\tNode_3_Val\t' \
                          'Label\tElement\n'
        else:
            print('Not implemented')
            return
        if elem_type == '2D':
            if nb_of_nodes ==8:
                elements = self.get_surface(xi = 'xi1',
                                        output_elem_type='quadrilateral')
            else:
                elements = self.elements
        elif elem_type == '3D':
            if nb_of_nodes ==8:
                elements = self.elements
            else:
                print('invalid elem type: the input elements are 3D elements')
        # Write nodes
        for indx, elem in enumerate(elements):
            for node_id in elem:
                elem_string += '%i\t' % (node_id+1)
            elem_string += '%i\t%i\n' % (indx + 1, indx + 1)

        f.write(elem_string)
        f.close()

    def export_mesh_cont6(self, filename, elem_type = '2D',scale = 1):
        filename_nodes = filename + '_nodes'
        filename_elem = filename + '_elem'

        self.export_nodes_cont6(filename_nodes,scale =scale)
        self.export_elements_cont6(filename_elem, elem_type = elem_type)
        print('Continuity mesh exported')

    def export_nodes_to_cont6_data(self, filename,scale = 1):
        # %% Write Nodes list in Continuity format

        points = self.nodes
        weights = np.ones(len(points))
        local_filename = filename+'_data.txt'
        print('Writing data form {0}'.format(local_filename))

        f = open(local_filename, 'w+')
        # Write headers
        node_string = 'Coords_1_val\tCoords_1_weight\t' \
                      'Coords_2_val\tCoords_2_weight\t' \
                      'Coords_3_val\tCoords_3_weight\tData\n'

        # Write nodes

        for i,node in enumerate(points):
            for coord in node:
                node_string += '%f\t%f\t' % (coord*scale,weights[i])
            node_string += '%i\n' % (i + 1)

        f.write(node_string)
        f.close()

    def get_volume(self):


        vertex = self.nodes
        surface_index = self.elements
        d = [0,0,0]

        volume = 0
        for facet in surface_index:
            a = np.asarray(vertex[facet[0]])
            b = np.asarray(vertex[facet[1]])
            c = np.asarray(vertex[facet[2]])
            #Volume of a tetrahedron == 1/6 * abs((AB crossed with AC) dotted with AD)
            bd = b - d
            cd = c - d
            Vtetr = (a[2] - d[2]) * (bd[0] * cd[1] - bd[1] * cd[0]) +\
            (a[1] - d[1]) * (bd[2] * cd[0] - bd[0] * cd[2]) + \
            (a[0] - d[0] ) * (bd[1] * cd[2] - bd[2] * cd[1])

            volume += Vtetr


        # Vtetr will return in mm^3 however CVI42 output uses ml; 1ml == 1000mm3
        return volume/6000

    def get_normals(self):
        # Create a zeroed array with the same type and shape as our vertices
        # i.e., per vertex normal
        vertices = deepcopy(self.nodes)
        faces = deepcopy(self.elements)
        norm = np.zeros(vertices.shape, dtype=vertices.dtype)
        # Create an indexed view into the vertex array using
        # the array of three indices for triangles
        tris = vertices[faces]
        # Calculate the normal for all the triangles,
        # by taking the cross product of the vectors v1-v0,
        # and v2-v0 in each triangle
        n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
        # n is now an array of normals per triangle.
        # The length of each normal is dependent the vertices,
        # we need to normalize these.
        n = normalize_v3(n)
        # now we have a normalized array of normals, one per triangle
        # But instead of one per triangle (i.e., flat shading),
        # we add to each vertex in that triangle,  the triangles' normal.
        # Multiple triangles would then contribute to every vertex,
        # so we need to normalize again afterwards.

        norm[faces[:, 0]] += n
        norm[faces[:, 1]] += n
        norm[faces[:, 2]] += n
        norm = normalize_v3(norm)
        return norm

    def get_curvature(self):
        normals = self.get_normals()
        edges = self.get_lines()
        vertices = self.nodes
        curvature = np.zeros(vertices.shape[0], dtype=vertices.dtype)

        p_edge = vertices[edges]
        n_edge = normals[edges]
        # Calculate the curvatue for all the edges,
        # curvature at each vertex = (n2-n1)*(p2-p1)/|p2-p1|^2
        e_curv =  [np.dot(a,b) for a,b in
                      zip((p_edge[:,0] - p_edge[:,1]),
                          (n_edge[:,0]- n_edge[:,1]))]
        e_curv =np.array(e_curv)
        e_curv /= np.linalg.norm(p_edge[:,0] - p_edge[:,1], axis= 1)
        # e_curv contains the curvature per line
        # the curvature at each vertex is computed as the mean
        # of the curvatures of incident edges

        curvature[edges[:,0]] += e_curv
        curvature[edges[:,1]] += e_curv
        unique, counts = np.unique(edges, return_counts=True)
        curvature[unique] = np.divide(curvature[unique],counts)
        return curvature

    def crop_mesh(self, p0, normal, reidex_verts = False):
        '''
        Crop a closed mesh using an plane defined by a point and a normal
        :param p0: array(3) point o the plane
        :param normal: array(3) vector defining the plane normal
        :return: new closed mesh
        '''

        facets = deepcopy(self.elements)
        verts = deepcopy(self.nodes)
        centered_vertex = verts - [list(p0)] * len(verts)

        distance_to_plane = np.dot(normal, centered_vertex.T)

        selected_verts, = np.where(np.sign(distance_to_plane)>=0)

        binary_facets = np.isin(facets, selected_verts)
        selected_facets = binary_facets.sum(axis =1)
        #  select all the facets have at least one node on the
        # side of interest



        rhs_com =facets[selected_facets == 3]
        lhs_com = facets[np.logical_not(selected_facets == 3)]

        out_meshes = self.break_mesh([rhs_com,lhs_com], reidex_verts = reidex_verts)

        return out_meshes

    def crop_by_landmarks(self,landmarks, reidex_verts = False):
        '''
        Crop a closed mesh be using the distance to the given landmarks to compute the
        components
        :param landmarks: (nx3) n landmarks
        :return: list of n meshes
        '''

        verts = deepcopy(self.nodes)
        facets = deepcopy(self.elements)
        landmarks_tree = cKDTree(landmarks)
        distance, closest_ld_index = landmarks_tree.query(verts)
        components = np.unique(closest_ld_index)


        com_facets = []
        for c in components:
            selected_verts_ids, = np.where(closest_ld_index == c)
            binary_facets = np.isin(facets, selected_verts_ids)
            selected_facets = binary_facets.sum(axis=1)
            com_facets.append( facets[selected_facets >1])


        return self.break_mesh(com_facets, reidex_verts = False)

    def break_mesh(self, components, reidex_verts = True):
        '''

        :param components: list of m  (nx3) arrays giving the subsets of faces for each new component,
                            m is the number of components
        :return: list of new closed meshes
        '''
        facets = deepcopy(self.elements)
        verts = deepcopy(self.nodes)
        out_meshes = []
        boundary_verts = np.empty([0,3])
        for com_facets in components:
            com_verts = np.unique(com_facets)
            # the new mesh is an open mesh, to fill in the hole
            # we will create an additional vertex computed as the centroid
            # of the boundary. Then create additional facets between the boundary
            # nodes and the centroid

            #  To select the boundary vertices, we compute the vertex connectivity
            #  in the full and new mesh.
            #  If the vertex connectivity has decreased in the new mesh comparing with the
            #  full mesh,
            #  then the vertex is a boundary node (at the cutting edge)

            conectivity_mask = np.array(
                [facets == x for x in com_verts])
            conectivity_mask = conectivity_mask.sum(axis=2)
            conectivity_mask = conectivity_mask.sum(axis=1)

            new_conectivity_mask = np.array(
                [com_facets == x for x in com_verts])
            new_conectivity_mask = new_conectivity_mask.sum(axis=2)
            new_conectivity_mask = new_conectivity_mask.sum(axis=1)
            boundary_verts_ids = np.array([com_verts[x] for x in range(len(
                com_verts)) if new_conectivity_mask[x] < conectivity_mask[x]])

            if reidex_verts:
                #  reindex nodes in the new croped mesh
                new_verts_id = np.array(range(len(com_verts)))
                for vert in new_verts_id:
                    com_facets[com_facets == com_verts[vert]] = vert
                    boundary_verts_ids[boundary_verts_ids == com_verts[vert]] = \
                        vert
            else:
                new_verts_id = np.array(range(self.nb_nodes))

            # select facets containing the boundary verts,
            # we will call them boundary facets
            boundary_neighbours = np.array([com_facets == x for x in
                                            boundary_verts_ids])

            # select boundary  facets (at the cutting edge) where all
            # 3 vertices are in the boundary vertex subgroub
            # in this case to new facets need to be created.
            # Example, let assume that vertex 1,3 are attached to
            # the new mesh, ut still defined as boundary.
            # 1 -_
            # |      -_
            # |          2
            # |     _-
            # 3 _-
            # new facets to create are (1,2,centroid) and (3,2,centroid)
            # vertex 2 will have the lower connectivity.
            fully_connected_facets = com_facets[boundary_neighbours.sum(axis=0).sum(axis=1) == 3]
            new_rhs_com = deepcopy(com_facets)
            materials = [0]*len(new_rhs_com)
            if len(fully_connected_facets) > 0:
                low_connectivity = boundary_verts_ids[
                    boundary_neighbours.sum(axis=1).sum(axis=1) == 1]
                new_facets = []
                for facet in fully_connected_facets:
                    changing_vertex = np.logical_not(np.isin(facet, low_connectivity))
                    changing_index = np.array(range(3))[changing_vertex]
                    for index in changing_index:
                        new_facet = deepcopy(facet)
                        new_facet[index] = new_verts_id.max() + 1
                        new_facets.append(new_facet)

                new_facets = np.array(new_facets)
                new_facets = np.array([new_facets[:, 0],
                                       new_facets[:, 2],
                                       new_facets[:, 1]]).T
                new_rhs_com = np.concatenate((new_rhs_com, new_facets))
                materials = materials+[1]*len(new_facets)

            # now select boundary facets with 2 vertices
            # Remark, in the previous example  facet (1,2,3) , the underling
            # facets (x, 1,3) contains 2 verts in the boundary subset, however is not
            # a boundary facet, therefore we need to exclude them
            new_facets = com_facets[boundary_neighbours.sum(axis=0).sum(axis=1) == 2]
            boundary_neighbours = boundary_neighbours.sum(axis=0)
            boundary_neighbours = boundary_neighbours[boundary_neighbours.sum(axis=1) == 2]
            inside_facets = np.isin(new_facets, np.unique(fully_connected_facets))
            inside_facets = inside_facets.sum(axis=1)
            new_facets = new_facets[np.logical_not(inside_facets == 2), :]
            boundary_neighbours = boundary_neighbours[np.logical_not(inside_facets == 2)]

            # for each boundary facet a new facet needs to be created between
            # the two verts from boundary subset linked to the centroid
            new_facets[boundary_neighbours[:, 0] == 0, 0] = new_verts_id.max() + 1
            new_facets[boundary_neighbours[:, 1] == 0, 1] = new_verts_id.max() + 1
            new_facets[boundary_neighbours[:, 2] == 0, 2] = new_verts_id.max() + 1

            new_facets = np.array([new_facets[:, 0],
                                   new_facets[:, 2],
                                   new_facets[:, 1]]).T

            new_rhs_com = np.concatenate((new_rhs_com, new_facets))
            materials = materials + [1]*len(new_facets)
            if reidex_verts:
                new_verts = verts[com_verts]

            else:
                new_verts = verts
            boundary_verts = np.concatenate((boundary_verts, new_verts[boundary_verts_ids]))

            new_mesh = Mesh('rhs_mesh')
            new_mesh.set_nodes(new_verts)
            new_mesh.set_elements(new_rhs_com)
            new_mesh.set_materials(list(range(len(new_rhs_com))),materials)
            out_meshes.append(deepcopy(new_mesh))

        centroid = np.median(boundary_verts, axis=0).reshape(1, -1)
        #centroid = np.mean(landmarks, axis=0).reshape(1, -1)

        for mesh in out_meshes:
            new_verts = np.concatenate((mesh.nodes, centroid))
            mesh.set_nodes(new_verts)

        return  out_meshes

    def get_intersection_with_plane(self, P0, N0):
        ''' Calculate intersection points between a plane with the mesh
            P = L.GetIntersectiontWithPlane(P0,N0,'opt1',val1,...)
            [P,Fidx] = L.GetIntersectionWithPlane(P0,N0,'opt1',val1,...)

            The plane is defined by the normal N0 and a point P0 on the plane.

            - P{i} are Nx3 coordinate points on surface i that intersect with the
              plane.
         '''

        # Adjust P0 & N0 into a column vector
        N0 = N0 / np.linalg.norm(N0)



        faces = deepcopy(self.elements)

        # --- find triangles that intersect with plane: (P0,N0)
        # calculate sign distance of each vertices

        # set the origin of the model at P0
        centered_vertex = self.nodes - [list(P0)]*len(self.nodes)
        # projection on the normal
        dist = np.dot(N0, centered_vertex.T)

        sgnDist = np.sign(dist[faces])
        # search for triangles having the vertex on the both sides of the
        # frame plane => intersecting with the frame plane
        valid_index = [np.any(sgnDist[i] > 0) and np.any(sgnDist[i] < 0)
                           for i in range(len(sgnDist))]
        intersecting_face_idx = np.where(valid_index)[0]

        if len(intersecting_face_idx) < 0:
            return np.empty((0,3))


        # Find the intersection lines - find segments for each intersected
        # triangles that intersects the plane

        iPos = [x for x in intersecting_face_idx if np.sum(sgnDist[x] >0) == 1] #all
        # triangles with one vertex on the positive part
        iNeg = [x for x in intersecting_face_idx if np.sum(sgnDist[x] <0) == 1] # all
        # triangles with one vertex on the negative part
        p1 = []
        u = []


        for face_index in iPos:  # triangles where only one
            # point on positive side
            # pivot points
            pivot_point_mask = sgnDist[face_index, :] > 0
            res =centered_vertex[faces[face_index, pivot_point_mask ], :][0]
            p1.append(list(res))
            # u vectors
            u = u + list( np.subtract(
                centered_vertex[faces[face_index,
                                      np.invert(pivot_point_mask)] ,:],
                [list(res)]*2))

        for face_index in iNeg:  # triangles where only one
            # point on negative side
            # pivot points
            pivot_point_mask = sgnDist[face_index, :] < 0
            res = centered_vertex[faces[face_index, pivot_point_mask] ,
                  :][0] # select the vertex on the negative side
            p1.append(res)
            # u vectors
            u = u + list(np.subtract(
                centered_vertex[faces[face_index,
                np.invert(pivot_point_mask)],:],
                [list(res)]*2)
            )


        # calculate the intersection point on each triangle side
        u = np.asarray(u).T
        p1 = np.asarray(p1).T
        if len(p1) == 0:
            return None

        mat = np.zeros((3, 2 * p1.shape[1]))
        mat[0:3, 0::2] = p1
        mat[0:3, 1::2] = p1
        p1 = mat
        sI = - np.dot(N0.T, p1) / (np.dot(N0.T, u))
        factor_u = np.array([list(sI)]*3)
        pts = np.add(p1, np.multiply(factor_u, u)).T
        # add vertices that are on the surface
        Pon = centered_vertex[faces[sgnDist == 0], :]
        pts = np.vstack((pts, Pon))
        # #change points to the original position
        Fidx = pts + [list(P0)]*len(pts)
        return Fidx





