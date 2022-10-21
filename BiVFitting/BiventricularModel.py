from scipy import spatial
import functools

import scipy
from copy import deepcopy

#local imports
from .GPDataSet import *
from .surface_enum import Surface
from .surface_enum import ContourType
from .surface_enum import SURFACE_CONTOUR_MAP
from .fitting_tools import *
from  .build_model_tools import *
from line_profiler import LineProfiler
from collections import OrderedDict
from nltk import flatten

##Author : Charlène Mauger, University of Auckland, c.mauger@auckland.ac.nz

class BiventricularModel():
    """ This class creates a surface from the control mesh, based on
    Catmull-Clark subdivision surface method. Surfaces have the following properties:

    Attributes:
       numNodes = 388                       Number of control nodes.
       numElements = 187                    Number of elements.
       numSurfaceNodes = 5810               Number of nodes after subdivision
                                            (surface points).
       control_mesh                         Array of x,y,z coordinates of
                                            control mesh (388x3).
       et_vertex_xi                         local xi position (xi1,xi2,xi3)
                                            for each vertex (5810x3).

       et_pos                               Array of x,y,z coordinates for each
                                            surface nodes (5810x3).
       et_vertex_element_num                Element num for each surface
                                            nodes (5810x1).
       et_indices                           Elements connectivities (n1,n2,n3)
                                            for each face (11760x3).
       basis_matrix                         Matrix (5810x388) containing basis
                                            functions used to evaluate surface
                                            at surface point locations
       matrix                               Subdivision matrix (388x5810).


       GTSTSG_x, GTSTSG_y, GTSTSG_z         Regularization/Smoothing matrices
                                            (388x388) along
                                            Xi1 (circumferential),
                                            Xi2 (longitudinal) and
                                            Xi3 (transmural) directions


       apex_index                           Vertex index of the apex

       et_vertex_start_end                  Surface index limits for vertices
                                            et_pos. Surfaces are sorted in
                                            the following order:
                                            LV_ENDOCARDIAL, RV septum, RV free wall,
                                            epicardium, mitral valve, aorta,
                                            tricuspid, pulmonary valve,
                                            RV insert.
                                            Valve centroids are always the last
                                            vertex of the corresponding surface

       surface_start_end                    Surface index limits for embedded
                                            triangles et_indices.
                                            Surfaces are sorted in the following
                                            order:  LV_ENDOCARDIAL, RV septum, RV free wall,
                                            epicardium, mitral valve, aorta,
                                            tricuspid, pulmonary valve, RV insert.

       mBder_dx, mBder_dy, mBder_dz         Matrices (5049x338) containing
                                            weights used to calculate gradients
                                            of the displacement field at Gauss
                                            point locations.

       Jac11, Jac12, Jac13                  Matrices (11968x388) containing
                                            weights used to calculate Jacobians
                                            at Gauss point location (11968x338).
                                            Each matrix element is a linear
                                            combination of the 388 control points.
                                            J11 contains the weight used to
                                            compute the derivatives along Xi1,
                                            J12 along Xi2 and J13 along Xi3.
                                            Jacobian determinant is
                                            calculated/checked on 11968 locations.
       fraction                 gives the level of the patch
                                (level 0 = 1,level 1 = 0.5,level 2 = 0.25)
       b_spline                  gives the 32 control points which need to be weighted
                                (for each vertex)
       patch_coordinates        patch coordinates
       boundary                 boundary
       phantom_points           Some elements only have an epi surface.
                                The phantomt points are 'fake' points on
                                the endo surface.


    """
    numNodes = 388
    '''Class constant, Number of control nodes (388).'''
    numElements = 187
    '''Class constant, number of elements (187).
 '''
    numSurfaceNodes = 5810
    '''class constant, number of nodes after subdivision
                                            (5810).'''
    apex_index = 50  # endo #5485 #epi
    '''class constant, vertex index defined as the apex point.'''

    et_vertex_start_end = np.array(
        [[0, 1499], [1500, 2164], [2165, 3223], [3224, 5581],
         [5582, 5630], [5631, 5655], [5656, 5696], [5697, 5729],
         [5730, 5809]])
    '''Class constant, surface index limits for vertices `et_pos`. 
    Surfaces are defined in the following order:
    
        LV_ENDOCARDIAL = 0 
        RV_SEPTUM = 1
        RV_FREEWALL = 2
        EPICARDIAL =3
        MITRAL_VALVE =4
        AORTA_VALVE = 5
        TRICUSPID_VALVE = 6
        PULMONARY_VALVE = 7
        RV_INSERT = 8
    
    For a valve surface the centroids are defined last point of the 
    corresponding surface. To get surface end and start vertex index use 
    get_surface_vertex_start_end_index(surface_name)
    
    Example
    --------
        lv_endo_start_idx= et_vertex_start_end[0][0]
        lv_endo_end_idx= et_vertex_start_end[0][1]
        lv_aorta_end_idx= et_vertex_start_end[5][1]-1
        lv_aorta_centroid_idx= et_vertex_start_end[5][1]
        
        lv_endo_start_idx,lv_endo_end_idx = 
        mesh.get_surface_vertex_start_end_index(surface_name)
       
    surface_name as defined in `Surface` class
    '''
    surface_start_end = np.array(
        [[0, 3071], [3072, 4479], [4480, 6751], [6752, 11615],
         [11616, 11663], [11664, 11687], [11688, 11727],
         [11728, 11759]])
    '''Class constant,  surface index limits for embedded triangles `et_indices`.
    Surfaces are defined in the following order  
      
            LV_ENDOCARDIAL = 0 
            RV_SEPTUM = 1
            RV_FREEWALL = 2
            EPICARDIAL =3
            MITRAL_VALVE =4
            AORTA_VALVE = 5
            TRICUSPID_VALVE = 6
            PULMONARY_VALVE = 7
            RV_INSERT = 8
    
    To get surface end and start vertex index use 
    get_surface_start_end_index(surface_name)
    
    Example
    --------
        lv_endo_start_idx= surface_start_end[0][0]
        lv_endo_end_idx= surface_start_end[0][1]
        lv_aorta_start_idx= surface_start_end[5][0]
        lv_aorta_end_idx= surface_start_end[5][1]
        
        lv_endo_start_idx,lv_endo_end_idx = 
        mesh.get_surface_start_end_index(surface_name)
    '''

    def __init__(self, control_mesh_dir, label = 'default', build_mode = False):
        """ Return a Surface object whose control mesh should be
            fitted to the dataset *DataSet*.

            control_mesh is always the same - this is the RVLV template. If
            the template is changed, one needs to regenerate all the matrices.
            The build_mode allows to load the data needed to interpolate a
            surface field. For fitting purposes set build_mode to False
        """
        self.build_mode = build_mode
        '''
        False by default, true to evaluate surface points at xi local 
        coordinates
        '''
        if not os.path.exists(control_mesh_dir):
            ValueError('Invalid directory name')

        self.label = label
        model_file = os.path.join(control_mesh_dir,"model.txt")
        if not os.path.exists(model_file):
            ValueError('Missing model.txt file')
        self.control_mesh = (pd.read_table
                             (model_file, delim_whitespace=True, header=None, engine = 'c')).values 

        ''' `numNodes`X3 array[float] of x,y,z coordinates of control mesh.
        '''

        subdivision_matrix_file = os.path.join(control_mesh_dir,
                                               "subdivision_matrix.txt")
        if not os.path.exists(subdivision_matrix_file):
            ValueError('Missing subdivision_matrix.txt')

        self.matrix = (pd.read_table(subdivision_matrix_file,
                                     delim_whitespace=True,
                                     header=None, engine = 'c')).values.astype(float)
        '''Subdivision matrix (`numNodes`x`numSurfaceNodes`).
        '''

        self.et_pos = np.dot(self.matrix, self.control_mesh)
        '''`numSurfaceNodes`x3 array[float] of x,y,z coordinates for each
                                            surface nodes.
        '''

        et_index_file = os.path.join(control_mesh_dir,'ETIndicesSorted.txt')
        if not os.path.exists(et_index_file):
            ValueError('Missing ETIndicesSorted.txt file')
        self.et_indices = (pd.read_table(et_index_file, delim_whitespace=True,
                                            header=None, engine = 'c')).values.astype(int)-1
        ''' 11760x3 array[int] of elements connectivity (n1,n2,n3) for each face.'''
        
        #et_index_thruWall_file = os.path.join(control_mesh_dir, 'ETIndicesThruWall.txt') #RB addition for MyoMass calc
        et_index_thruWall_file = os.path.join(control_mesh_dir, 'epi_to_septum_ETindices.txt')
        if not os.path.exists(et_index_thruWall_file):
            ValueError('Missing ETIndicesThruWall.txt file for myocardial mass calculation')
        self.et_indices_thruWall = (
            pd.read_table(et_index_thruWall_file, delim_whitespace=True,
                          header=None)).values.astype(int)-1

        et_index_EpiLVRV_file = os.path.join(control_mesh_dir, 'ETIndicesEpiRVLV.txt') #RB addition for MyoMass calc
        if not os.path.exists(et_index_EpiLVRV_file):
            ValueError('Missing ETIndicesEpiRVLV.txt file for myocardial mass calculation')
        self.et_indices_EpiLVRV = (
            pd.read_table(et_index_EpiLVRV_file, delim_whitespace=True,
                          header=None, engine = 'c')).values.astype(int)-1


        GTSTSG_x_file = os.path.join(control_mesh_dir,'GTSTG_x.txt')
        if not os.path.exists(GTSTSG_x_file):
            ValueError(' Missing GTSTG_x.txt file')
        self.GTSTSG_x = (
            pd.read_table(GTSTSG_x_file, delim_whitespace=True,
                          header=None, engine = 'c')).values.astype(float)
        '''`numNodes`x`numNodes` Regularization/Smoothing matrix along Xi1 (
        circumferential direction)        
        '''

        GTSTSG_y_file = os.path.join(control_mesh_dir,'GTSTG_y.txt')
        if not os.path.exists(GTSTSG_y_file):
            ValueError(' Missing GTSTG_y.txt file')
        self.GTSTSG_y = (
            pd.read_table(GTSTSG_y_file, delim_whitespace=True,
                          header=None, engine = 'c')).values.astype(float)
        '''`numNodes`x`numNodes` Regularization/Smoothing matrix along
                                            Xi2 (longitudinal) direction'''

        GTSTSG_z_file = os.path.join(control_mesh_dir,'GTSTG_z.txt')
        if not os.path.exists(GTSTSG_z_file):
            ValueError(' Missing GTSTG_z.txt file')
        self.GTSTSG_z = (
            pd.read_table(GTSTSG_z_file, delim_whitespace=True,
                          header=None, engine = 'c')).values.astype(float)
        '''`numNodes`x`numNodes` Regularization/Smoothing matrix along
                                                    Xi3 (transmural) direction'''

        etVertexElementNum_file = os.path.join(control_mesh_dir,
                                               'etVertexElementNum.txt')
        if not os.path.exists(etVertexElementNum_file):
            ValueError('Missing etVertexElementNum.txt file')
        self.et_vertex_element_num = \
            (pd.read_table(etVertexElementNum_file,
                           delim_whitespace=True,header=None, engine = 'c')).values[:,0].astype(
                int)-1

        '''`numSurfaceNodes`x1 array[int] Element num for each surface nodes.
        Used for surface evaluation 
        '''

        mBder_x_file = os.path.join(control_mesh_dir,'mBder_x.txt')
        if not os.path.exists(mBder_x_file):
            ValueError('Missing mBder_x.file')
        self.mBder_dx = (
            pd.read_table(mBder_x_file, delim_whitespace=True,
                          header=None, engine = 'c')).values.astype(float)
        '''`numSurfaceNodes`x`numNodes` Matrix containing  weights used to 
        calculate gradients of the displacement field at Gauss point locations.
        '''
        mBder_y_file = os.path.join(control_mesh_dir,'mBder_y.txt')
        if not os.path.exists(mBder_y_file):
            ValueError('Missing mBder_y.file')
        self.mBder_dy = (
            pd.read_table(mBder_y_file, delim_whitespace=True,
                          header=None, engine = 'c')).values.astype(float)
        '''`numSurfaceNodes`x`numNodes` Matrix containing  weights used to 
        calculate gradients of the displacement field at Gauss point locations.
        '''

        mBder_z_file = os.path.join(control_mesh_dir,'mBder_z.txt')
        if not os.path.exists(mBder_z_file):
            ValueError('Missing mBder_z.file')
        self.mBder_dz = (
            pd.read_table(mBder_z_file, delim_whitespace=True,
                          header=None, engine = 'c')).values.astype(float)
        '''`numSurfaceNodes`x`numNodes` Matrix containing  weights used to 
        calculate gradients of the displacement field at Gauss point locations.
        '''

        jac11_file = os.path.join(control_mesh_dir,'J11.txt')
        if not os.path.exists(jac11_file):
            ValueError('Missing J11.txt file')

        self.Jac11 = (pd.read_table(jac11_file, delim_whitespace=True,
                                    header=None, engine = 'c')).values.astype(float)
        '''11968 x `numNodes` matrix containing weights used to calculate 
        Jacobians  along Xi1 at Gauss point location.
        Each matrix element is a linear combination of the 388 control points.
    
        '''

        jac12_file = os.path.join(control_mesh_dir, 'J12.txt')
        if not os.path.exists(jac12_file):
            ValueError('Missing J12.txt file')

        self.Jac12 = (pd.read_table(jac12_file, delim_whitespace=True,
                                    header=None, engine = 'c')).values.astype(float)
        '''11968 x `numNodes` matrix containing weights used to calculate 
        Jacobians  along Xi2 at Gauss point location.
        Each matrix element is a linear combination of the 388 control points.
        '''
        jac13_file = os.path.join(control_mesh_dir, 'J13.txt')
        if not os.path.exists(jac13_file):
            ValueError('Missing J13.txt file')

        self.Jac13 = (pd.read_table(jac13_file, delim_whitespace=True,
                                    header=None, engine = 'c')).values.astype(float)
        '''11968 x `numNodes` matrix containing weights used to calculate 
        Jacobians along Xi3 direction at Gauss point location.
        Each matrix element is a linear combination of the 388 control points.

        '''
        basic_matrix_file = os.path.join(control_mesh_dir,'basis_matrix.txt')
        if not os.path.exists(basic_matrix_file):
            ValueError('Missing basis_matrix.txt file')
        self.basis_matrix = (pd.read_table(basic_matrix_file,
                          delim_whitespace=True,header=None, engine = 'c')).values.astype(
            float)  #
        '''`numSurfaceNodes`x`numNodes` array[float]  basis  functions used 
        to evaluate surface at surface point locations
        '''

        if not self.build_mode:
            return


        et_vertex_xi_file = os.path.join(control_mesh_dir,"etVertexXi.txt")
        if not os.path.exists(et_vertex_xi_file):
            ValueError('Missing etVertexXi.txt file')
        self.et_vertex_xi = (pd.read_table(
            et_vertex_xi_file, delim_whitespace=True, header=None, engine = 'c')).values
        ''' `numSurfaceNodes`x3 array[float] of local xi position (xi1,xi2,
        xi3)
                                        for each vertex.
        '''

        b_spline_file = os.path.join(control_mesh_dir, "control_points_patches.txt")
        if not os.path.exists(b_spline_file):
            ValueError('Missing control_points_patches.txt file')
        self.b_spline = (pd.read_table(
            b_spline_file, delim_whitespace=True, header=None, engine = 'c')).values.astype(int)-1
        ''' numSurfaceNodesX32 array[int] of 32 control points which need to be 
         weighted (for each vertex)
        '''
        boundary_file = os.path.join(control_mesh_dir,"boundary.txt")
        if not os.path.exists(boundary_file):
            ValueError('Missing boundary.txt file')
        self.boundary = (pd.read_table(
            boundary_file, delim_whitespace=True, header=None, engine = 'c')).values.astype(int)
        ''' boundary'''

        control_ef_file = os.path.join(control_mesh_dir,
                                       "control_mesh_connectivity.txt")
        if not os.path.exists(control_ef_file):
            ValueError('Missing control_mesh_connectivity.txt file')
        self.control_et_indices = (pd.read_table(
            control_ef_file, delim_whitespace=True, header=None, engine = 'c')).values.astype(int)-1
        ''' (K,8) matrix of control mesh connectivity'''

        phantom_points_file = os.path.join(control_mesh_dir, "phantom_points.txt")
        if not os.path.exists(phantom_points_file):
            ValueError('Missing phantom_points.txt file')
        self.phantom_points = (pd.read_table(
            phantom_points_file, delim_whitespace=True, header=None, engine = 'c')).values.astype(float)
        ''' Some surface nodes are not needed for the 
        definition of the biventricular 2D surface therefore they are 
        not include in the surface node matrix. However they are 
        necessary for the 3D interpolation (septum area).
        these elements are called the phantom points and the 
        corresponding information as the subdivision level , local
        patch coordinates etc. are stored in phantom points array
        '''
        self.phantom_points[:,:17] = self.phantom_points[:,:17].astype(int)-1
        patch_coordinates_file = os.path.join(control_mesh_dir,
                                              "patch_coordinates.txt")
        if not os.path.exists(patch_coordinates_file):
            ValueError('Missing patch_coordinates.txt file')
        self.patch_coordinates = (pd.read_table(
            patch_coordinates_file, delim_whitespace=True, header=None, engine = 'c')).values
        '''local patch coordinates. 
         
        According to CC subdivision surface, to evaluate a point on a surface 
        the original control mesh needs to be subdivided in 'child' patches.  
        
        The coordinates of the child patches are then used to map the local 
        coordinates with respect to control mesh in to the local 
        coordinates with respect to child patch.
        
        The patch coordinates and subdivision level of each surface node are 
        pre-computed and here imported as patch_coordinates and fraction.
        
        For details see
        
        Atlas-based Analysis of Biventricular Heart 
        Shape and Motion in Congenital Heart Disease. C. Mauger (p34-37)
        '''
        fraction_file = os.path.join(control_mesh_dir, "fraction.txt")
        if not os.path.exists(fraction_file):
            ValueError('Missing fraction.txt file')
        self.fraction = (pd.read_table(
            fraction_file, delim_whitespace=True, header=None, engine = 'c')).values
        '''`numSurfaceNodes`x1 vector[int] subdivision level of the 
         patch (level 0 = 1,level 1 = 0.5,level 2 = 0.25). See 
         `patch_coordinates for details`
        '''

        local_matrix_file = os.path.join(control_mesh_dir,
                                         "local_matrix.txt")

        if not os.path.exists(local_matrix_file):
            ValueError('Missing local_matrix.txt file')
        self.local_matrix = (pd.read_table(
            local_matrix_file, delim_whitespace=True, header=None, engine = 'c')).values

    def get_nodes(self):
        '''

        Returns
        --------
        `numSurfaceNodes`x3 array of vertices coordinates
        '''
        return self.et_pos
    def get_control_mesh_nodes(self):
        '''

        Returns
        -------
        `numNodes`x3 array of coordinates of control points

        '''
        return self.control_mesh
    def get_surface_vertex_start_end_index(self, surface_name):
        """ Return first and last vertex index for a given surface to use 
        with `et_pos` array
        
        Parameters
        -----------

        `surface_name`  Surface name as defined in 'Surface' enumeration

        `Returns`
        ---------
        2x1 array with first and last vertices index belonging to
            surface_name
        """


        if surface_name == Surface.LV_ENDOCARDIAL:
            return self.et_vertex_start_end[0, :]

        if surface_name == Surface.RV_SEPTUM:
            return self.et_vertex_start_end[1, :]

        if surface_name == Surface.RV_FREEWALL:
            return self.et_vertex_start_end[2, :]

        if surface_name == Surface.EPICARDIAL:
            return self.et_vertex_start_end[3, :]

        if surface_name == Surface.MITRAL_VALVE:
            return self.et_vertex_start_end[4, :]

        if surface_name == Surface.AORTA_VALVE:
            return self.et_vertex_start_end[5, :]

        if surface_name == Surface.TRICUSPID_VALVE:
            return self.et_vertex_start_end[6, :]

        if surface_name == Surface.PULMONARY_VALVE:
            return self.et_vertex_start_end[7, :]

        if surface_name == Surface.RV_INSERT:
            return self.et_vertex_start_end[8, :]
        if surface_name == Surface.APEX:
            return [self.apex_index]*2

    def get_surface_start_end_index(self, surface_name):
        """  Return first and last element index for a given surface, tu use 
        with `et_indices` array 
        
        
        Parameters
        ----------

        `surface_name` surface name as defined by `Surface` enum

        Returns
        -------

        2x1 array containing first and last vertices index belonging to
           `surface_name`
        """

        if surface_name == Surface.LV_ENDOCARDIAL:
            return self.surface_start_end[0, :]

        if surface_name == Surface.RV_SEPTUM:
            return self.surface_start_end[1, :]

        if surface_name == Surface.RV_FREEWALL:
            return self.surface_start_end[2, :]

        if surface_name == Surface.EPICARDIAL:
            return self.surface_start_end[3, :]

        if surface_name == Surface.MITRAL_VALVE:
            return self.surface_start_end[4, :]

        if surface_name == Surface.AORTA_VALVE:
            return self.surface_start_end[5, :]

        if surface_name == Surface.TRICUSPID_VALVE:
            return self.surface_start_end[6, :]

        if surface_name == Surface.PULMONARY_VALVE:
            return self.surface_start_end[7, :]


    def is_diffeomorphic(self, new_control_mesh, min_jacobian):
        """ This function checks the Jacobian value at Gauss point location
        (I am using 3x3x3 per element).
        
        Notes
        ------
        Returns 0 if one of the determinants is below a given
        threshold and 1 otherwise.
        It is recommended to use min_jacobian = 0.1 to make sure that there 
        is no intersection/folding; a value of 0 can be used, but it might 
        still give a positive jacobian
        if there are small intersections due to numerical approximation.

        Parameters
        -----------

        `new_control_mesh` control mesh we want to check

        Returns
        -------

        'min_jacobian' float Jacobian threshold

        """

        boolean = 1
        for i in range(len(self.Jac11)):

            jacobi = np.array(
                [[np.inner(self.Jac11[i, :], new_control_mesh[:, 0]),
                  np.inner(self.Jac12[i, :], new_control_mesh[:, 0]),
                  np.inner(self.Jac13[i, :], new_control_mesh[:, 0])],
                 [np.inner(self.Jac11[i, :], new_control_mesh[:, 1]),
                  np.inner(self.Jac12[i, :], new_control_mesh[:, 1]),
                  np.inner(self.Jac13[i, :], new_control_mesh[:, 1])],
                 [np.inner(self.Jac11[i, :], new_control_mesh[:, 2]),
                  np.inner(self.Jac12[i, :], new_control_mesh[:, 2]),
                  np.inner(self.Jac13[i, :], new_control_mesh[:, 2])]])


            determinant = np.linalg.det(jacobi)
            if determinant < min_jacobian:
                boolean = 0
                return boolean

        return boolean

    def CreateNextModel(self, DataSetES, ESTranslation):
        """Copy of the current model onto the next time model.

        Parameters
        ----------

        `DataSetES` dataset for the new time frame

        `ESTranslation` 2D translations needed

        Returns
        --------

        `ESSurface` Copy of the current model (`biv_model`),
                associated with the new DataSet.
        """

        ESSurface = copy.deepcopy(self)
        ESSurface.data_set = copy.deepcopy(DataSetES)
        ESSurface.SliceShiftES(ESTranslation, self.image_position_patient)

        return ESSurface

    def update_pose_and_scale(self,dataset):
        """ A method that scale and translate the model to rigidly align 
        with the guide points.

        Notes
        ------

        Parameters
        ------------
        
        `dataset` GPDataSet object with guide points
                
        Returns
        -------- 
        
        `scaleFactor` scale factor between template and data points.
             """

        scale_factor = self._get_scaling(dataset)
        self.update_control_mesh(self.control_mesh * scale_factor)

        # The rotation is defined about the origin so we need to translate the model to the origin
        self.update_control_mesh(self.control_mesh - self.et_pos.mean(axis=0))
        rotation = self._get_rotation(dataset)
        self.update_control_mesh( np.array([np.dot(rotation, node) for node in
                                      self.control_mesh]))

        # Translate the model back to origin of the DataSet coordinate system

        translation = self._get_translation(dataset)
        self.update_control_mesh(self.control_mesh+translation)

 # et_pos update

        return scale_factor

    def _get_scaling(self, dataset):
        '''Calculates a scaling factor for the model
        to match the guide points defined in datset

        Parameters
        -----------

        `data_set` GPDataSet object

        Returns
        --------

        `scaleFactor` float
        '''
        model_shape_index = [self.apex_index,
                       self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1],
                       self.get_surface_vertex_start_end_index(Surface.TRICUSPID_VALVE)[1]]
        model_shape = np.array( self.et_pos[model_shape_index,:])
        reference_shape = np.array([dataset.apex,
                           dataset.mitral_centroid,
                           dataset.tricuspid_centroid])
        mean_m = model_shape.mean(axis = 0)
        mean_r = reference_shape.mean(axis = 0)
        model_shape = model_shape -mean_m
        reference_shape = reference_shape - mean_r
        ss_model = (model_shape**2).sum()
        ss_reference = (reference_shape**2).sum()
        #centered Forbidius norm
        norm_model = np.sqrt(ss_model)
        reference_norm = np.sqrt(ss_reference)


        scaleFactor = reference_norm/norm_model

        return scaleFactor

    def _get_translation(self, dataset):
        ''' Calculates a translation for (x, y, z)
        axis that aligns the model RV center with dataset RV center

        Parameters
        -----------
        `data_set` GPDataSet object

        Returns
        --------
          `translation` 3X1 array[float] with x, y and z translation

        '''
        t_points_index_1 =  (dataset.contour_type ==
                            ContourType.SAX_RV_FREEWALL) | (
                         dataset.contour_type == ContourType.SAX_RV_FREEWALL) | (
                         dataset.contour_type == ContourType.PULMONARY_VALVE) | (
                         dataset.contour_type == ContourType.TRICUSPID_VALVE)
        t_points_index_2 = (dataset.contour_type ==ContourType.SAX_LV_ENDOCARDIAL) | \
                           (dataset.contour_type == ContourType.SAX_LV_ENDOCARDIAL) | (
                           dataset.contour_type == ContourType.MITRAL_VALVE) | (
                           dataset.contour_type == ContourType.AORTA_VALVE)
        # t_points_index = np.logical_or (dataset.contour_type == ContourType.SAX_RV_EPICARDIAL ,
        #                                 dataset.contour_type == ContourType.SAX_RV_SEPTUM)
        points_coordinates_1 = dataset.points_coordinates[t_points_index_1]
        points_coordinates_2 = dataset.points_coordinates[t_points_index_2]
        translation = (points_coordinates_1.mean(
            axis=0)+points_coordinates_2.mean(axis=0))*0.5
        return translation

    def  _get_rotation(self, data_set):
        '''Computes the rotation between model and data set,
        the rotation is given
        by considering the x-axis direction defined by the mitral valve centroid
        and apex the origin of the coordinates system is the mid point between
        the apex and mitral centroid

        Parameters
        ----------

        `data_set` GPDataSet object

        Returns
        --------
        `rotation` 3x3 rotation matrix
        '''


        base = data_set.mitral_centroid

        # computes data_set coordinates system
        xaxis = data_set.apex - base
        xaxis = xaxis / np.linalg.norm(xaxis)

        apex_position_model = self.et_pos[self.apex_index, :]
        base_model = self.et_pos[
                     self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1], :]

        xaxis_model = apex_position_model- base_model
        xaxis_model = xaxis_model / np.linalg.norm(xaxis_model) #normalize

        # compute origin defined at 1/3 of the height of the model on the Ox
        # axis
        tempOrigin = 0.5 * (data_set.apex + base)
        tempOrigin_model = 0.5*(apex_position_model +base_model)

        maxd = np.linalg.norm(0.5 * (data_set.apex - base))
        mind = -np.linalg.norm(0.5 * (data_set.apex - base))

        maxd_model = np.linalg.norm(0.5 * (apex_position_model- base_model))
        mind_model = -np.linalg.norm(0.5 * (apex_position_model - base_model))

        point_proj = data_set.points_coordinates[(data_set.contour_type ==
                                                  ContourType.LAX_LV_ENDOCARDIAL), :]
        #point_proj = np.vstack((point_proj,data_set.points_coordinates[
        #                       (data_set.contour_type == ContourType.LAX_RV_FREEWALL), :]))
        #point_proj = np.vstack((point_proj,data_set.points_coordinates[
        #                       (data_set.contour_type == ContourType.SAX_RV_SEPTUM), :]))
        #point_proj = np.vstack((point_proj,data_set.points_coordinates[
        #                       (data_set.contour_type == ContourType.LAX_RV_SEPTUM), :]))

        #point_proj = np.vstack((point_proj,
        #                        data_set.points_coordinates[
        #                        ( data_set.contour_type == ContourType.SAX_RV_FREEWALL), :]))
        point_proj = np.vstack((point_proj,
                                data_set.points_coordinates[
                                ( data_set.contour_type == ContourType.LAX_LV_ENDOCARDIAL), :]))

        if len(point_proj) == 0:
            point_proj = np.vstack((point_proj,data_set.points_coordinates[
                                  (data_set.contour_type == ContourType.SAX_RV_SEPTUM), :]))
            point_proj = np.vstack((point_proj,
                                   data_set.points_coordinates[
                                   ( data_set.contour_type == ContourType.SAX_RV_FREEWALL), :]))

        if len(point_proj) == 0:
            ValueError('Missing contours in update_pose_and_scale')
            return


        tempd = [np.dot(xaxis, p) for p in (point_proj - tempOrigin)]
        maxd = max(np.max(tempd), maxd)
        mind = min(np.min(tempd), mind)

        model_epi = self.et_pos[self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[0]:
                               self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[1]+ 1,:]
        tempd_model = [np.dot(xaxis_model, point_model)
                       for point_model in (model_epi- tempOrigin_model)]
        maxd_model = max(np.max(tempd_model), maxd_model)
        mind_model = min(np.min(tempd_model), mind_model)

        centroid = tempOrigin + mind * xaxis  + ((maxd - mind) / 3.0) * xaxis
        centroid_model = tempOrigin_model + mind_model * xaxis_model + \
                         ((maxd_model - mind_model) / 3.0) * xaxis_model


        #Compute Oy axis
        valid_index = (data_set.contour_type == ContourType.SAX_RV_FREEWALL) + \
                      (data_set.contour_type == ContourType.SAX_RV_SEPTUM)
        rv_endo_points = data_set.points_coordinates[ valid_index, :]

        rv_points_model = \
            self.et_pos[ self.get_surface_vertex_start_end_index(Surface.RV_SEPTUM)[0]:
                         self.get_surface_vertex_start_end_index(Surface.RV_FREEWALL)[1] + 1, :]

        rv_centroid = rv_endo_points.mean(axis=0)
        rv_centroid_model = rv_points_model.mean(axis=0)

        scale = np.dot(xaxis, rv_centroid) - np.dot(xaxis, centroid)/np.dot(xaxis,xaxis)
        scale_model = np.dot(xaxis_model, rv_centroid_model) - np.dot(xaxis_model,
                     centroid_model)/np.dot(xaxis_model,xaxis_model)
        rvproj = centroid + scale * xaxis
        rvproj_model = centroid_model + scale_model * xaxis_model


        yaxis = rv_centroid - rvproj
        yaxis_model = rv_centroid_model - rvproj_model

        yaxis = yaxis / np.linalg.norm(yaxis)
        yaxis_model = yaxis_model / np.linalg.norm(yaxis_model)

        zaxis = np.cross(xaxis, yaxis)
        zaxis_model = np.cross(xaxis_model, yaxis_model)

        zaxis = zaxis / np.linalg.norm(zaxis)
        zaxis_model = zaxis_model / np.linalg.norm(zaxis_model)

        # Find translation and rotation between the two coordinates systems
        # The easiest way to solve it (in my opinion) is by using a
        #Singular Value Decomposition as reported by Markley (1988):
        #    1. Obtain a matrix B as follows:
        #        B=∑ni=1aiwiviTB=∑i=wiviT
        #    2. Find the SVD of BB
        #        B=USVT
        #    3. The rotation matrix is:
        #        R=UMVT, where M=diag([11det(U)det(V)])
        #

        # Step 1
        B = np.outer(xaxis, xaxis_model) \
            + np.outer(yaxis, yaxis_model) \
            + np.outer(zaxis, zaxis_model)

        # Step 2
        [U, s, Vt] = np.linalg.svd(B)

        M = np.array([[1, 0, 0], [0, 1, 0],
                      [0, 0, np.linalg.det(U) * np.linalg.det(Vt)]])
        Rotation = np.dot(U, np.dot(M, Vt))

        return Rotation

    def plot_surface(self, face_color_LV, face_color_RV, face_color_epi,
                     surface="all", opacity=0.8):
        """ Plot 3D model.

        Parameters
        -----------

        `face_color_LV` LV_ENDOCARDIAL surface color

        `face_color_RV` RV surface color

        `face_color_epi` Epicardial color

        `surface` surface to plot, default all = entire surface,
                    endo = endocardium, epi = epicardium

        `opacity` float surface opacity

        Returns
        --------

        `triangles_epi` Nx3 array[int] triangles defining the epicardium surface

        `triangles_LV` Nx3 array[int] triangles defining the LV endocardium
        surface

        `triangles_RV` Nx3 array[int]  triangles defining the RV surface

        `lines` lines that need to be plotted
        """

        x = np.array(self.et_pos[:, 0]).T
        y = np.array(self.et_pos[:, 1]).T
        z = np.array(self.et_pos[:, 2]).T

        # LV_ENDOCARDIAL endo
        surface_index = self.get_surface_start_end_index(Surface.LV_ENDOCARDIAL)
        I_LV = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1, 0])
        J_LV = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,1])
        K_LV = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1] + 1,2])
        
        simplices_lv = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        # RV free wall
        surface_index = self.get_surface_start_end_index(Surface.RV_FREEWALL)
        I_FW = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,
                          0] )
        J_FW = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,
                          1] )
        K_FW = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,
                          2] )
        simplices_fw = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        # RV septum
        surface_index = self.get_surface_start_end_index(Surface.RV_SEPTUM)
        I_S = np.asarray(self.et_indices[
                         surface_index[0]:surface_index[1]+1,
                         0] )
        J_S = np.asarray(self.et_indices[
                         surface_index[0]:surface_index[1]+1,
                         1] )
        K_S = np.asarray(self.et_indices[
                         surface_index[0]:surface_index[1]+1,
                         2] )
        simplices_s = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        # Epicardium
        surface_index = self.get_surface_start_end_index(Surface.EPICARDIAL)
        I_epi = np.asarray(self.et_indices[
                           surface_index[0]:surface_index[1]+1, 0] )
        J_epi = np.asarray(self.et_indices[
                           surface_index[0]:surface_index[1]+1, 1] )
        K_epi = np.asarray(self.et_indices[
                           surface_index[0]:surface_index[1]+1, 2] )
        simplices_epi = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        if surface == "all":
            points3D = np.vstack(
                (self.et_pos[:, 0], self.et_pos[:, 1], self.et_pos[:, 2])).T

            tri_vertices_epi = list(map(lambda index: points3D[index], simplices_epi))
            tri_vertices_fw = list(
                map(lambda index: points3D[index], simplices_fw))
            tri_vertices_s = list(
                map(lambda index: points3D[index], simplices_s))
            tri_vertices_lv= list(
                map(lambda index: points3D[index], simplices_lv))

            triangles_LV = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_LV,
                i=I_LV,
                j=J_LV,
                k=K_LV,
                opacity=1,
                name = 'LV edocardial',
                showlegend = True,
            )

            triangles_FW = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_RV,
                name='RV free wall',
                showlegend=True,
                i=I_FW,
                j=J_FW,
                k=K_FW,
                opacity=1
            )

            triangles_S = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_RV,
                name='RV septum',
                showlegend=True,
                i=I_S,
                j=J_S,
                k=K_S,
                opacity=1
            )

            triangles_epi = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_epi,
                i=I_epi,
                j=J_epi,
                k=K_epi,
                opacity=0.4,
                name = 'epicardial',
                showlegend=True,
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_epi]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_epi = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe epicardial'
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_fw]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_fw = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe rv free wall'
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_s]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_s = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe rv septum'
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_lv]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_lv = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe lv edocardial'
            )

            return [triangles_epi, triangles_LV, triangles_FW,triangles_S,
                    lines_epi, lines_lv,lines_fw,lines_s]

        if surface == "endo":
            points3D = np.vstack(
                (self.et_pos[:, 0], self.et_pos[:, 1], self.et_pos[:, 2])).T
            simplices = np.vstack((self.et_indices[
                                   self.surface_start_end[0, 0]:
                                   self.surface_start_end[2, 1]+1, 0] ,
                                   self.et_indices[
                                   self.surface_start_end[0, 0]:
                                   self.surface_start_end[2, 1]+1,1] ,
                                   self.et_indices[
                                   self.surface_start_end[0, 0]:
                                   self.surface_start_end[2,1]+1,2] )).T

            tri_vertices = list(map(lambda index: points3D[index], simplices))

            triangles_LV = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_LV,
                i=I_LV,
                j=J_LV,
                k=K_LV,
                opacity=opacity,
                name = 'LV edocardial',
                showlegend=True,
            )

            triangles_FW = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_RV,
                name='RV freewall',
                showlegend=True,
                i=I_FW,
                j=J_FW,
                k=K_FW,
                opacity=opacity)

            # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
            lists_coord = [
                [[T[k % 3][c] for k in range(4)] for T in tri_vertices] for c in
                range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,
                name = 'wireframe'
            )

            return triangles_LV, triangles_FW, lines

        if surface == "epi":
            surface_index = self.get_surface_start_end_index(Surface.EPICARDIAL)
            points3D = np.vstack(
                (self.et_pos[:, 0], self.et_pos[:, 1], self.et_pos[:, 2])).T
            simplices = np.vstack((self.et_indices[surface_index[ 0]:
                                                   surface_index[1]+1,0] ,
                                   self.et_indices[
                                   surface_index[ 0]:
                                   surface_index[ 1]+1,1] ,
                                   self.et_indices[surface_index[ 0]:
                                                   surface_index[ 1]+1,2] )).T

            tri_vertices = list(map(lambda index: points3D[index], simplices))

            triangles_epi = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_epi,
                i=I_epi,
                j=J_epi,
                k=K_epi,
                opacity=0.8,
                name = 'epicardial',
                showlegend=True
            )

            # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
            lists_coord = [
                [[T[k % 3][c] for k in range(4)] for T in tri_vertices] for c in
                range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                name = 'wireframe',
                showlegend=True
            )

            return [triangles_epi, lines]

    def PlotSurface(self, face_color_LV, face_color_RV, face_color_epi, my_name,
                    surface="all", opacity=0.8):
        """ Plot 3D model.
            Input:
               face_color_LV, face_color_RV, face_color_epi: LV_ENDOCARDIAL, RV and epi colors
               my_name: surface name
               surface (optional): all = entire surface,
               endo = endocardium, epi = epicardium  (default = "all")
            Output:
               triangles_epi, triangles_LV, triangles_RV: triangles that
               need to be plotted for the epicardium, LV_ENDOCARDIAL and Rv respectively
               lines: lines that need to be plotted
        """

        x = np.array(self.et_pos[:, 0]).T
        y = np.array(self.et_pos[:, 1]).T
        z = np.array(self.et_pos[:, 2]).T

        # LV_ENDOCARDIAL endo
        surface_index = self.get_surface_start_end_index(Surface.LV_ENDOCARDIAL)
        I_LV = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1, 0])
        J_LV = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,1])
        K_LV = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1] + 1,2])
        simplices_lv = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        # RV free wall
        surface_index = self.get_surface_start_end_index(Surface.RV_FREEWALL)
        I_FW = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,
                          0] )
        J_FW = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,
                          1] )
        K_FW = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,
                          2] )
        simplices_fw = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        # RV septum
        surface_index = self.get_surface_start_end_index(Surface.RV_SEPTUM)
        I_S = np.asarray(self.et_indices[
                         surface_index[0]:surface_index[1]+1,
                         0] )
        J_S = np.asarray(self.et_indices[
                         surface_index[0]:surface_index[1]+1,
                         1] )
        K_S = np.asarray(self.et_indices[
                         surface_index[0]:surface_index[1]+1,
                         2] )
        simplices_s = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        # Epicardium
        surface_index = self.get_surface_start_end_index(Surface.EPICARDIAL)
        I_epi = np.asarray(self.et_indices[
                           surface_index[0]:surface_index[1]+1, 0] )
        J_epi = np.asarray(self.et_indices[
                           surface_index[0]:surface_index[1]+1, 1] )
        K_epi = np.asarray(self.et_indices[
                           surface_index[0]:surface_index[1]+1, 2] )
        simplices_epi = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        if surface == "all":
            points3D = np.vstack(
                (self.et_pos[:, 0], self.et_pos[:, 1], self.et_pos[:, 2])).T

            tri_vertices_epi = list(map(lambda index: points3D[index], simplices_epi))
            tri_vertices_fw = list(
                map(lambda index: points3D[index], simplices_fw))
            tri_vertices_s = list(
                map(lambda index: points3D[index], simplices_s))
            tri_vertices_lv= list(
                map(lambda index: points3D[index], simplices_lv))

            triangles_LV = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_LV,
                i=I_LV,
                j=J_LV,
                k=K_LV,
                opacity=1,
                name = 'LV edocardial',
                showlegend = True,
            )

            triangles_FW = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_RV,
                name='RV free wall',
                showlegend=True,
                i=I_FW,
                j=J_FW,
                k=K_FW,
                opacity=1
            )

            triangles_S = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_RV,
                name='RV septum',
                showlegend=True,
                i=I_S,
                j=J_S,
                k=K_S,
                opacity=1
            )

            triangles_epi = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_epi,
                i=I_epi,
                j=J_epi,
                k=K_epi,
                opacity=0.4,
                name = 'epicardial',
                showlegend=True,
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_epi]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_epi = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe epicardial'
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_fw]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_fw = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe rv free wall'
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_s]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_s = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe rv septum'
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_lv]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_lv = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe lv edocardial'
            )

            return [triangles_epi, triangles_LV, triangles_FW,triangles_S,
                    lines_epi, lines_lv,lines_fw,lines_s]

        if surface == "endo":
            points3D = np.vstack(
                (self.et_pos[:, 0], self.et_pos[:, 1], self.et_pos[:, 2])).T
            simplices = np.vstack((self.et_indices[
                                   self.surface_start_end[0, 0]:
                                   self.surface_start_end[2, 1]+1, 0] ,
                                   self.et_indices[
                                   self.surface_start_end[0, 0]:
                                   self.surface_start_end[2, 1]+1,1] ,
                                   self.et_indices[
                                   self.surface_start_end[0, 0]:
                                   self.surface_start_end[2,1]+1,2] )).T

            tri_vertices = list(map(lambda index: points3D[index], simplices))

            triangles_LV = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_LV,
                i=I_LV,
                j=J_LV,
                k=K_LV,
                opacity=opacity,
                name = 'LV edocardial',
                showlegend=True,
            )

            triangles_FW = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_RV,
                name='RV freewall',
                showlegend=True,
                i=I_FW,
                j=J_FW,
                k=K_FW,
                opacity=opacity)

            triangles_S = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_RV,
                name='RV septum',
                showlegend=True,
                i=I_S,
                j=J_S,
                k=K_S,
                opacity=opacity)
            # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
            lists_coord = [
                [[T[k % 3][c] for k in range(4)] for T in tri_vertices] for c in
                range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,
                name = 'wireframe'
            )

            return [triangles_LV, triangles_FW, lines]

        if surface == "epi":
            surface_index = self.get_surface_start_end_index(Surface.EPICARDIAL)
            points3D = np.vstack(
                (self.et_pos[:, 0], self.et_pos[:, 1], self.et_pos[:, 2])).T
            simplices = np.vstack((self.et_indices[surface_index[ 0]:
                                                   surface_index[1]+1,0] ,
                                   self.et_indices[
                                   surface_index[ 0]:
                                   surface_index[ 1]+1,1] ,
                                   self.et_indices[surface_index[ 0]:
                                                   surface_index[ 1]+1,2] )).T

            tri_vertices = list(map(lambda index: points3D[index], simplices))

            triangles_epi = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_epi,
                i=I_epi,
                j=J_epi,
                k=K_epi,
                opacity=0.8,
                name = 'epicardial',
                showlegend=True
            )

            # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
            lists_coord = [
                [[T[k % 3][c] for k in range(4)] for T in tri_vertices] for c in
                range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                name = 'wireframe',
                showlegend=True
            )

            return [triangles_epi, lines]



    def get_intersection_with_plane(self, P0, N0, surface_to_use = None):
        ''' Calculate intersection points between a plane with the
        biventricular model (LV_ENDOCARDIAL only)

        Parameters
        ----------
        `P0` (3,1) array[float] a point of the plane
        `N0` (3,1) array[float normal to the plane

        Returns
        -------

        `Fidx` (N,3) array[float] are indices of the surface nodes indicating
        intersecting the plane'''

        # Adjust P0 & N0 into a column vector
        N0 = N0 / np.linalg.norm(N0)

        Fidx = []

        if surface_to_use is None:
            surface_to_use = [Surface.LV_ENDOCARDIAL]
        for surface in surface_to_use:  # We just want intersection LV_ENDOCARDIAL,
            # RVS. RVFW, epi
            # Get the faces
            faces = self.get_surface_faces(surface)

            # --- find triangles that intersect with plane: (P0,N0)
            # calculate sign distance of each vertices

            # set the origin of the model at P0
            centered_vertex = self.et_pos - [list(P0)]*len(self.et_pos)
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
            # see http://softsurfer.com/Archive/algorithm_0104/algorithm_0104B.htm

            # pivot points

            iPos = [x for x in intersecting_face_idx if np.sum(sgnDist[x] >0) == 1] #all
            # triangles with one vertex on the positive part
            iNeg = [x for x in intersecting_face_idx if np.sum(sgnDist[x] <0) == 1] # all
            # triangles with one vertex on the negative part
            p1 = []
            u = []


            for face_index in iPos:  # triangles where only one
                # point
                # on positive side
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
                continue
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
            Fidx = Fidx + list(pts + [list(P0)]*len(pts))

        return Fidx

    def get_intersection_with_dicom_image(self, frame,surface =None):
        ''' Get the intersection contour points between the biventricular
        model with a DICOM image

        Example
        -------
            obj.get_intersection_with_dicom_image(frame, Surface.RV_SEPTUM)

        Parameters
        ----------

        `frame` Frame obj with the dicom information

        `surface` Surface enum, model surface to be intersected

        Returns
        -------

        `P` (n,3) array[float] intersecting points
        '''

        image_position = np.asarray(frame.position,
                                    dtype=float)
        image_orientation = np.asarray(frame.orientation,
                                       dtype=float)


        # get image position and the image vectors
        V1 = np.asarray(image_orientation[0:3], dtype=float)
        V2 = np.asarray(image_orientation[3:6], dtype=float)
        V3 = np.cross(V1, V2)


        # get intersection points
        P = self.get_intersection_with_plane(image_position, V3, surface_to_use=surface)

        return P

    def get_surface_faces(self, surface):
        ''' Get the faces definition for a surface triangulation'''

        surface_index = self.get_surface_start_end_index(surface)
        return self.et_indices[ surface_index[0]:surface_index[1] + 1, :]


    def compute_data_xi(self, weight, data):
        """Projects the N guide points to the closest point of the model
        surface.

        If 2 data points are projected  onto the same surface point,
        the closest one is kept. Surface type is matched with the Contour
        type using 'SURFACE_CONTOUR_MAP' variable (surface_enum)

        Parameters
        -----------

        `weight` float with weights given to the data points

        `data` GPDataSet object with guide points

        Returns
        --------

        `data_points_index` (`N`,1) array[int] with index of the closest
        control point to the each node


        `w` (`N`,`N`) matrix[float] of  weights of the data points


        `distance_d_prior` (`N`,1) matrix[float] distances to the closest points

        `psi_matrix` basis function matrix (`N`,`numNodes`)

        """

        data_points = np.array(data.points_coordinates)
        data_contour_type = np.array(data.contour_type)
        data_weights = np.array(data.weights)
        psi_matrix = []
        w = []
        distance_d_prior = []
        index = []
        data_points_index = []
        indexL = []
        index_unique = [] #add by LDT 3/11/2021

        basis_matrix = self.basis_matrix

        # add by A. Mira : a more compressed way of initializing the cKDTree

        for surface in Surface:
            # Trees initialization
            surface_index = self.get_surface_vertex_start_end_index(
                surface)
            tree_points = self.et_pos[
                          surface_index[0]:surface_index[1] + 1, :]
            if len(tree_points) == 0:
                continue
            surface_tree = scipy.spatial.cKDTree(tree_points)

            # loop over contours is faster, for the same contours we are using
            # the same tree, therefore the query operation can be done for all
            # points of the same contour: A. Mira 02/2020
            for contour in SURFACE_CONTOUR_MAP[surface.value]:
                contour_points_index = np.where(data_contour_type == contour)[0]
                contour_points = data_points[contour_points_index]
                
                #if np.isnan(np.sum(contour_points))==True:
                    #LDT 7/03: handle error, why do I get nan in contours? 
                    #continue

                weights_gp = data_weights[contour_points_index]

                if len(contour_points) == 0:
                    continue
                
                if surface.value < 4:  # these are the surfaces
                    
                    distance, vertex_index = surface_tree.query(contour_points, k=1, p=2)
                    index_closest = [x + surface_index[0] for x in vertex_index]
                    #add by LDT (3/11/2021): perform preliminary operations for vertex points that are not in index
                    # instead of doing them in the 'else' below. This makes the for loop below faster.
                    unique_index_closest = list(OrderedDict.fromkeys(index_closest))    #creates a list of elements that are unique in index_closest
                    dict_unique = dict(zip(unique_index_closest, range(0,len(unique_index_closest))))   #create a dictionary = {'unique element': its list index}
                    vrtx = list(dict_unique.keys())                     # list of all the dictionary keys 
                    common_elm = list(set(index_unique) & set(vrtx))    # intersection between the array index_unique and the unique points in index_closest
                  
                    def filter_new(full_list, excludes):
                        '''
                        eliminates the items in 'exclude' out of the full_list
                        '''
                        s = set(excludes)
                        return (x for x in full_list if x not in s)

                    # togli gli elementi comuni 
                    index_unique.append(list(filter_new(vrtx, common_elm))) # stores the new vertices that are NOT in already in the index_unique list
                    index_unique = flatten(index_unique)

                    items_as_dict = dict(zip(index_unique,range(0,len(index_unique)))) # builds a dictionary = {vertices: indexes}

                    for i_idx, vertex_index in enumerate(index_closest):
                        if len(set([vertex_index]).intersection(index))==0: #changed by LDT (3/11/2021): faster
                            index.append(int(vertex_index))
                            data_points_index.append(contour_points_index[i_idx])
                            psi_matrix.append(basis_matrix[int(vertex_index), :])
                            w.append(weight * weights_gp[i_idx])
                            distance_d_prior.append(distance[i_idx])

                        else:
                            old_idx = items_as_dict[vertex_index] #changed by LDT (3/11/2021)
                            distance_old = distance_d_prior[old_idx]
                            if distance[i_idx] < distance_old:
                                distance_d_prior[old_idx] = distance[i_idx]
                                data_points_index[old_idx] = \
                                    contour_points_index[i_idx]
                                w[old_idx] = weight * weights_gp[i_idx]

                
                else:
                  
                    # If it is a valve, we virtually translate the data points
                    # (only the ones belonging to the same surface) so their centroid
                    # matches the template's valve centroid.
                    # So instead of calculating the minimum distance between the point
                    # p and the model points pm, we calculate the minimum distance between
                    # the point p+t and pm,
                    # where t is the translation needed to match both centroids
                    # This is to make sure that the data points are going to be
                    # projected all around the valve and not only on one side.
                    if surface.value < 8:  # these are the landmarks without apex
                      
                        # and rv inserts
                        centroid_valve = self.et_pos[surface_index[1]]
                        centroid_GP_valve = contour_points.mean(axis=0)
                        translation_GP_model = centroid_valve - centroid_GP_valve
                        translated_points = np.add(contour_points,
                                                   translation_GP_model)
            

                    else:  # rv_inserts  and apex don't
                        # need to be translated
                        translated_points = contour_points

                    if contour in [ContourType.MITRAL_PHANTOM,
                                   ContourType.PULMONARY_PHANTOM,
                                   ContourType.AORTA_PHANTOM,
                                   ContourType.TRICUSPID_PHANTOM]:
                                     
                        surface_tree = scipy.spatial.cKDTree(translated_points)
                        tree_points = tree_points[:-1]
                        distance, vertex_index = surface_tree.query(tree_points, k=1, p=2)
                        index_closest = [x + surface_index[0] for x in
                                         range(len(tree_points))]
                        weights_gp = [weights_gp[x] for x in vertex_index]

                        contour_points_index = [contour_points_index[x] for x
                                                in vertex_index]

                    else:
                       
                        distance, vertex_index = surface_tree.query(
                            translated_points, k=1, p=2)
                        index_closest = []
                        for x in vertex_index:
                            if (x + surface_index[0]) != surface_index[1]:
                                index_closest.append(x + surface_index[0])
                            else:
                                index_closest.append(x + surface_index[0] - 1)
                    

                    index = index + index_closest
                    psi_matrix = psi_matrix + list(basis_matrix[
                                                   index_closest, :])

                    w = w + [(weight * x) for x in weights_gp]
                    distance_d_prior = distance_d_prior + list(distance)
                    data_points_index = data_points_index + list(
                        contour_points_index)

        return [np.asarray(data_points_index), np.asarray(w),
                np.asarray(distance_d_prior), np.asarray(psi_matrix)]

    def extract_linear_hex_mesh(self, reorder_nodes=True):
        """
        Compute linear hex mesh associated with control mesh topology using
        points position from the subdivision surface.

        The new position is mapped using the nodes local coordinates (within
        element) from the subdivision surface mesh. The nodes of the new
        hex mesh will take the corner position of the corresponding
        control element (where xi are filed with 0 and 1 only).  The new
        control mesh will interpolate the subdivision surface
        at local coordinates (0,0,0),(1,0,0),(0,1,0),(1,1,0),
        (0,0,1), (1,0,1),(0,1,1),(1,1,1).

        Parameters:
        -----------

        `reorder_nodes' if true the nodes ids are reindexed

        Returns
        --------

        `new_nodes_position` (numNodes,3) array[float] new position of the nodes

        `new_elements` (nbElem, 8) array mesh connectivity

        """

        new_elements = np.zeros_like(self.control_et_indices)
        if reorder_nodes:
            new_nodes_position = np.zeros_like(self.control_mesh)
            nodes_id = np.sort(np.unique(self.control_et_indices))
        else:
            new_nodes_position = np.zeros_like(self.et_pos)
        #node_maping = np.zeros(mesh.et_pos.shape[0])
        xi_order = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        node_elem_map = self.et_vertex_element_num
        node_position = self.et_pos
        xi_position=self.et_vertex_xi


        for elem_id, elem_nodes in enumerate(self.control_et_indices):
            #
            elem_index = np.where(np.equal(node_elem_map, elem_id))[0]
            elem_index = elem_index.reshape(elem_index.shape[0])
            enode_position = node_position[elem_index]
            enode_xi = xi_position[elem_index, :]

            for node_index, node in enumerate(elem_nodes):

                index = np.prod(enode_xi == xi_order[node_index],
                                axis=1).astype(bool)
                if index.any():
                    if reorder_nodes:
                        new_node_id  = np.where(nodes_id == node)[0][0]
                    else:
                        new_node_id = node
                    new_nodes_position[new_node_id] = enode_position[index][0]
                    elem_index = self.control_et_indices==node
                    new_elements[elem_index] = new_node_id
                    #node_maping[node] = elem_index[index][0]

        return new_nodes_position, new_elements

    def evaluate_derivatives(self,xi_position,elements=None):
        '''
        Evaluate derivatives at xi_position within each element in *elements* list'

        Parameters:
        -----------

        xi_position (n,3) array of xi positions were to estimate derivatives

        elements: (m) list of elements were to estimate derivatives


        Returns:
        --------

        deriv (n,9): du, dv, duv, duu, dvv, dw, dvw, duw, dww, dudvdw
        '''

        if not self.build_mode:
            print('Derivatives are computed in build mode '
                  'build model with build_mode=True')
            return

        if elements is None:
            elements = list(range(np.max(self.et_vertex_element_num) + 1))

        num_points = len(elements)*len(xi_position)
        # interpolate field on surface nodes


        der_coeff = np.zeros((num_points, self.numNodes, 10))

        der_e, der_xi = list(),list()

        for et_index in elements:

            for j, xi in enumerate(xi_position):
                xig_index = et_index * len(xi_position) + j

                _, der_coeff[xig_index, :, :], _ = self.evaluate_basis_matrix(
                    xi[0],xi[ 1],  xi[2], et_index,  0,0,0)

                der_e.append(et_index)
                der_xi.append(xi)

        bxe = np.zeros((num_points, 10))
        bye = np.zeros((num_points, 10))
        bze = np.zeros((num_points, 10))

        for i in range(10):

            bxe[:,i] = np.dot(der_coeff [:,:,i], self.control_mesh[:,0])
            bye[:, i] = np.dot(der_coeff[:,:,i], self.control_mesh[:, 1])
            bze[:, i] = np.dot(der_coeff[:,:,i], self.control_mesh[:, 2])

        return der_e, der_xi, bxe,bye,bze

    def evaluate_basis_matrix(self, s, t, u, elem_number,
                              displacement_s=0,
                              displacement_t=0,
                              displacement_u=0):
        '''Evaluates position  and derivatives coefficients (basis function)
        matrix for a  local coordinate(s, t, u)
        (it is not a surface point) for the element elem_number.

        Notes
        ------
        Global coordinates of the point is obtained by dot product between
        position coefficients and control matrix

        Global surface derivatives at the given point is given by the dot
        product of the derivatives coefficients and the control matrix

        Parameters
        -----------

        `s` float between 0 and 1 local coordinate

        `t` float between 0 and 1 local coordinate

        `u` float between 0 and 1 local coordinate

        `elem_number` int element number in the control mesh
        (coarse mesh 186 elements )

        `displacement_s` float for finite difference calculation only (D-affine
        reg)

        `displacement_t` float for finite difference calculation only (D-affine
        reg)

        `displacement_u` float for finite difference calculation only (D-affine
        reg)


        Returns
        --------

        `full_matrix_coefficient_points` rows = number of data points
                                        columns = basis functions
                                        size = number of data points x 16

        `full_matrix_coefficient_der` (i, j, k) basis function for ith data
        point, jth basis fn, kth derivatives; size = number of  data  points
        x 16 x 5:  du, dv, duv, duu, dvv, dw, dvw, duw, dww, dudvdw

        `control_points` 16 control points B-spline


        '''
        if not self.build_mode:
            print('To evaluate gauss points the model should be '
                  'read with build_mode=True')
            return

        # Allocate memory
        params_per_element = 32
        pWeights = np.zeros((params_per_element)) # weights
        dWeights = np.zeros((params_per_element, 10)) # weights derivatives

        matrix_coefficient_Bspline_points = np.zeros(params_per_element)
        matrix_coefficient_Bspline_der = np.zeros((params_per_element, 10))
        full_matrix_coefficient_points = np.zeros(( self.numNodes))
        full_matrix_coefficient_der = np.zeros(( self.numNodes, 10))

        # The b-spline weight of a point is computed giving local coordinates
        # with respect to the 'child patches' elements. The input local
        # coordinates are given with respect to the control grid element

        # Local coordinates of the 'child' patches  are constant and
        # therefore they were precomputed and stored in patch_coordinates
        # and 'fraction` files. They will be later used to interpolate
        # 'patch' local coordinates (within surface element)
        # form 'face' local coordinates (within control grid element)


        #Find projection into endo and epi surfaces

        ps = np.zeros(2) #local coordinate in the child patch
        pt =np.zeros(2) # local coordinate in the child patch
        fract = np.zeros(2)
        boundary_value = np.zeros(2)
        b_spline_support = np.ones((2,self.b_spline.shape[1]))
        # select surface vertices associated with the given element number (
        # control mesh)
        # The element number is defined in the coarse mesh
        index_verts = self.et_vertex_element_num[:]==elem_number

        for surface in range(2):# s= 0 for endo surface and s = 1 for epi
            # surface
            # select vertices from the surface
            index_surface = self.et_vertex_xi[:,2] == surface
            element_verts_xi = self.et_vertex_xi[np.logical_and(index_surface, index_verts),:2]


            if len(element_verts_xi) > 0:
                # find the closest surface point
                elem_tree = cKDTree(element_verts_xi)
                ditance,closest_vertex_id = elem_tree.query([s,t])
                index_surface = np.where(
                    np.logical_and( index_surface, index_verts))[0][closest_vertex_id]

                #translate face to patch coordinates

                if self.fraction[index_surface] != 0:
                    ps[surface] = (s + displacement_s -
                                   element_verts_xi[closest_vertex_id, 0]) / \
                                  self.fraction[ index_surface]+  \
                                  self.patch_coordinates[index_surface, 0]
                    pt[surface] = (t + displacement_t -
                                   element_verts_xi[closest_vertex_id, 1]) / \
                                  self.fraction[index_surface]+ \
                                  self.patch_coordinates[index_surface,1]
                    b_spline_support[surface] = self.b_spline[index_surface, :]

                    boundary_value[surface] = self.boundary[index_surface]

                    fract[surface] = 1 / self.fraction[index_surface]

            elif elem_number >166:
                # some surface nodes are not needed for the
                # definition of the biventricular 2D surface therefore they are
                # not include in the surface node matrix. However they are
                # necessary for the 3D interpolation (septum area).
                # these elements are called the phantom points and the
                # corresponding information as the sudivision level ,
                # patch coordinates etc are stored in phantom points array.
                index_phantom = self.phantom_points[:, 0] == elem_number
                elem_phantom_points = self.phantom_points[index_phantom, :]
                elem_vertex_xi = np.stack((elem_phantom_points[:, 21],
                                       elem_phantom_points[:, 22])).T

                elem_tree = cKDTree(elem_vertex_xi)
                ditance, closest_vertex_id = elem_tree.query([s, t])

                if elem_phantom_points[closest_vertex_id, 24] != 0:
                    boundary_value[surface] = elem_phantom_points[closest_vertex_id, 17]
                    fract[surface] = 1 / elem_phantom_points[closest_vertex_id, 24]

                    ps[surface] = (s + displacement_s -
                                   elem_phantom_points[closest_vertex_id, 21]) / \
                         elem_phantom_points[closest_vertex_id, 24]+\
                    elem_phantom_points[closest_vertex_id, 18]
                    pt[surface] = (t + displacement_t -
                                   elem_phantom_points[closest_vertex_id, 22]) / \
                         elem_phantom_points[closest_vertex_id, 24]+\
                        elem_phantom_points[closest_vertex_id, 19]
                    b_spline_support[surface] = elem_phantom_points[closest_vertex_id, 1:17].astype(int)



        u1 = u + displacement_u
        # normalize s, t coordinates
        control_points = np.concatenate((b_spline_support[0],
                                         b_spline_support[1]))
        if len(control_points)<32:
            print('stop')
        # Uniform B - Splines basis functions
        sWeights = np.zeros((4,2))
        tWeights = np.zeros((4,2))
        uWeights = np.zeros(2)
        # Derivatives of the B - Splines basis functions
        ds = np.zeros((4, 2))
        dt = np.zeros((4, 2))
        du = np.zeros(2)
        # Second derivatives of the B - Splines basis functions
        dss = np.zeros((4,2))
        dtt = np.zeros((4,2))

        #populate arrays
        for surface in range(2):
            sWeights[:, surface] = basis_function_bspline(ps[surface])
            tWeights[:, surface] = basis_function_bspline(pt[surface])

            ds[:, surface] = der_basis_function_bspline(ps[surface])
            dt[:, surface] = der_basis_function_bspline(pt[surface])

            dss[:, surface ] = der2_basis_function_bspline(ps[surface])
            dtt[:, surface ] = der2_basis_function_bspline(pt[surface])

            # Adjust the boundaries
            sWeights[:, surface], tWeights[:, surface] = \
                adjust_boundary_weights(boundary_value[surface],
                sWeights[:, surface], tWeights[:, surface])

            ds[:, surface], dt[:, surface] = adjust_boundary_weights(
                boundary_value[surface],  ds[:, surface], dt[:, surface])

            dss[:, surface], dtt[:, surface] = adjust_boundary_weights(
                boundary_value[surface],dss[:, surface], dtt[:, surface])


        uWeights[0] = 1 - u1 # linear interpolation
        uWeights[1] = u1    # linear interpolation

        du[0] = -1
        du[1] = 1


        # Weights of the 16 tensors B - spline basis functions and their derivatives
        for k in range(2):
            for i in range(4):
                for j in range(4):
                    index = 16 * k + 4 * i + j
                    pWeights[index ] =  sWeights[j ,k] * tWeights[i ,k] * uWeights[ k ]

                    dWeights[index, 0] = ds[j, k ] * tWeights[i , k ] \
                                         * fract[k] * uWeights[k]

                    # dScale; % dphi / du = 2 ^ (p * n) * dx, where
                    #  n = level of the patch(0, 1 or 2) and p = order of differentiation.Here
                    #  p = 1 and n = 1 / biv_model.fraction(indx)
                    dWeights[index, 1] = sWeights[j, k] * dt[i,k] * fract[k] * uWeights[k]

                    dWeights[index, 2] = ds[j, k] * dt[i,k] * \
                                         (fract[k] ** 2) * uWeights[k ]

                    dWeights[index, 3] = dss[j, k] * tWeights[i,k] *  (fract[k] ** 2) * uWeights[k ]

                    dWeights[index, 4] = sWeights[j, k] * dtt[i,k] * \
                                         (fract[k] ** 2) * uWeights[k]

                    dWeights[index, 5] = sWeights[j, k] * tWeights[i,k] * du[k]

                    dWeights[index, 6] = sWeights[j, k] * dt[i,k] * du[k] * fract[k ]

                    dWeights[index, 7] = ds[j, k] * tWeights[i,k] * du[ k ] * fract[k]

                    dWeights[index, 8] = 0 #% linear interpolation --> duu = 0
                    dWeights[index, 9] = ds[j,k] * dt[i,k] * (fract[k]**2) * du[k]


            # add weights
        for i in range(32):
            matrix_coefficient_Bspline_points[i] = pWeights[i]
            full_matrix_coefficient_points = full_matrix_coefficient_points + pWeights[i] * \
                                             self.local_matrix[int(control_points[i]),:]
            for k in range(10):
                matrix_coefficient_Bspline_der[i, k] = dWeights[i, k]
                full_matrix_coefficient_der[:, k] = \
                    full_matrix_coefficient_der[:, k] + dWeights[i, k] * \
                    self.local_matrix[int(control_points[i]),:]

        return full_matrix_coefficient_points, \
               full_matrix_coefficient_der, \
               control_points

    def compute_local_cs(self, position, element_number=None):
        '''Computes local coordinates system at any point of the subdivision
        surface. x1 - circumferential direction, x2- longitudinal direction,
        x3- transmural direction

        Parameters
        ------------

        `position` list of (3,1) arrays[float] with xi coordinates

        `element_number` index of the control elements (coarse mesh). if non
        is specified the local cs is computed for all elements

        Return
        -------

        '''
        #todo method not tested
        if not self.build_mode:
            print('field evaluation is performed in build mode '
                  'build model with build_mode=True')
            return

        if element_number is None:
            element_number = np.array(range(np.max(self.et_vertex_element_num)))

        dxi = 0.01

        basis_matrix = np.zeros((
            len(element_number)*len(position), self.numNodes))
        basis_matrix_dx1 = np.zeros((
            len(element_number) * len(position), self.numNodes))
        basis_matrix_dx2 = np.zeros((
            len(element_number) * len(position), self.numNodes))
        basis_matrix_dx3 = np.zeros((
            len(element_number) * len(position), self.numNodes))
        for et_indx,control_et in enumerate(element_number):

            for j, xi in enumerate(position):
                g_indx = et_indx*len(position)+j
                # basis matrix for node position
                basis_matrix[g_indx,:], _, _ = self.evaluate_basis_matrix(
                    xi[0],  xi[1], xi[2], control_et, 0, 0, 0)

                # basis matrix for node position with increment of dxi in x1
                # direction
                basis_matrix_dx1[g_indx,:], _, _ = self.evaluate_basis_matrix(
                    xi[0],  xi[1], xi[2], control_et, dxi, 0, 0)

                # basis matrix for node position with increment of dxi in x2
                # direction
                basis_matrix_dx2[g_indx, :], _, _ = self.evaluate_basis_matrix(
                    xi[0], xi[1], xi[2], control_et, 0, dxi, 0)

                # basis matrix for node position with increment of dxi in x1
                # direction
                basis_matrix_dx3[g_indx, :], _, _ = self.evaluate_basis_matrix(
                    xi[0], xi[1], xi[2], control_et, 0, 0, dxi)

                #todo The local cs is computed by 1) dot product with control
                # matrix to compute node position. 2) subtract points to
                # compute the corresponding vectors



    def evaluate_surface_field(self, field,vertex_map):
        '''Evaluate field at the each surface points giving a sparse field
        defined at a subset of surface points.

        Notes
        ------

        Input surface field need to have more than 388 points evenly spread
        on the subdivided surface. A good choice is to define the filed at
        each surface point with each xi coordinate equal to 0 or 1

        Parameters
        -----------

        `field` (numNodes, k) array[float] field to be interpolated

        `vertex_map` (numNodes,1) array[ints] nodes id where the field is
        defined

        Returns
        --------

        `interpolated_field` (numSurfaceNodes, k) array[float] interpolated
        field at each surface node

        '''
        if not self.build_mode:
            print('field evaluation is performed in build mode '
                  'build model with build_mode=True')
            return


        basis_matrix = np.zeros((len(vertex_map),self.numNodes))

        # first estimate the control points from the known fiels
        for v_index,v in enumerate(vertex_map):
            xi = self.et_vertex_xi[v]
            et_index = self.et_vertex_element_num[v]

            basis_matrix[v_index,:], _, _ = self.evaluate_basis_matrix(
                    xi[0],  xi[1], xi[2], et_index, 0, 0, 0)

        control_points =  np.linalg.solve(basis_matrix,field)

        # interpolate field on surface nodes
        basis_matrix = np.zeros((self.numSurfaceNodes, self.numNodes))
        for et_index in range(np.max(self.et_vertex_element_num)+1):
            xig_index = np.where(self.et_vertex_element_num == et_index)[0]
            xig = self.et_vertex_xi[xig_index]
            for j, xi in enumerate(xig):
                basis_matrix[xig_index[j],:], _, _ = self.evaluate_basis_matrix(
                    xi[0],  xi[1], xi[2], et_index, 0, 0, 0)

        interpolated_field = np.dot(basis_matrix,control_points)


        return interpolated_field

    def evaluate_field(self,field, vertex_map, position, elements = None):
        '''Evaluates field at the each xi position within a element of the
        control grid

        Notes
        ------

        Input surface field need to have more than 388 points evenly spread
        on the subdivided surface. A good choice is to define the filed at
        each surface point with each xi coordinate equal to 0 or 1

        Parameters
        -----------

        `field` (`numNodes`,k) matrix[float] field to be interpolated

        `vertex_map` ('numNodes`) vector[ints] nodes id where the field is
        defined

        `position` (m,3) array[float] xi position where the field need to
        be interpolated

        `elements` list[int] list of elements (control grid) where the field
        need to be interpolated. If not specified the field is interfolated
        for all elements

        Returns
        ---------

        `interpolated_field` (N, k) matrix[float] interpolated field at each
        point. Where N = len(`elements')*len(position) and k=field.shape[1]

        `interpolated_points` (N,3) matrix[float] interpolated field at each
        point. Where N = len(`elements')*len(position)

        '''
        if not self.build_mode:
            print('field evaluation is performed in build mode '
                  'build model with build_mode=True')
            return
        if elements is None:
            elements = list(range(np.max(self.et_vertex_element_num)+1))

        basis_matrix = np.zeros((len(vertex_map),self.numNodes))

        # first estimate the control points from the known fiels
        for v_index,v in enumerate(vertex_map):
            xi = self.et_vertex_xi[v]
            et_index = self.et_vertex_element_num[v]

            basis_matrix[v_index,:], _, _ = self.evaluate_basis_matrix(
                    xi[0],  xi[1], xi[2], et_index, 0, 0, 0)

        control_points =  np.linalg.solve(basis_matrix,field)

        # interpolate field on surface nodes

        basis_matrix = np.zeros((len(elements)*len(position), self.numNodes))
        for i,et_index in enumerate(elements):
            for j, xi in enumerate(position):
                index = i*len(position)+j
                basis_matrix[index,:], _, _ = self.evaluate_basis_matrix(
                    xi[0],  xi[1], xi[2], et_index, 0, 0, 0)


        interpolated_field = np.dot(basis_matrix,control_points)
        points = np.dot(basis_matrix, self.control_mesh)

        return points,interpolated_field

    @staticmethod
    def get_tetrahedron_vol_CM( a, b, c, d):
        '''Calculates volume of tetrahedron abcd, where abc is the three
        surface point vertices and d is the fixed centroid for the shape
        Utilised in Mass/volume calculations where it is returned in ml3 once divided by 1000

        Parameters
        ----------
        `b`,  `c`, `a`, `d` (3,1) array tetrahedron vertices

        Returns
        --------
        float tetrahedron volume
        '''

        bxdx = b[0] - d[0]
        bydy = b[1] - d[1]
        bzdz = b[2] - d[2]
        cxdx = c[0] - d[0]
        cydy = c[1] - d[1]
        czdz = c[2] - d[2]

        vol = ((a[2] - d[2]) * ((bxdx*cydy) - (bydy*cxdx))) +\
              ((a[1] - d[1]) * ((bzdz*cxdx) - (bxdx*czdz))) +\
              ((a[0] - d[0]) * ((bydy*czdz) - (bzdz*cydy)))
        vol = vol/6
        return vol

    def get_ventricular_vol(self):
        '''Calculates the ventricular volumes of the left and right
        ventricles in ml3 .


        Parameters
        -----------


        Returns
        --------

        `LVvol` float ventricular volumes of the left ventricle in ml3

        `RVvol` float ventricular volumes of the right ventricles in ml3

        '''
        #
        #utput
        x = np.array(self.et_pos[:, 0]).T
        y = np.array(self.et_pos[:, 1]).T
        z = np.array(self.et_pos[:, 2]).T
        LvSurfaces = [Surface.LV_ENDOCARDIAL,
                      Surface.MITRAL_VALVE,
                      Surface.AORTA_VALVE] #surfaces that comprise LV closed surface
        RvSurfaces = [Surface.RV_FREEWALL,
                      Surface.TRICUSPID_VALVE,
                      Surface.PULMONARY_VALVE] #surfaces that comprise RV closed surface


        D = self.Get_centroid(isZero=False,
                                  I_EPI=np.asarray(
                                      self.et_indices[
                                      self.get_surface_vertex_start_end_index(
                                          Surface.LV_ENDOCARDIAL)[0]:
                                      self.get_surface_vertex_start_end_index(
                                          Surface.LV_ENDOCARDIAL)[1], 0]),
                              J_EPI=np.asarray(
                                  self.et_indices[
                                  self.get_surface_vertex_start_end_index(
                                      Surface.LV_ENDOCARDIAL)[0]:
                                  self.get_surface_vertex_start_end_index(
                                      Surface.LV_ENDOCARDIAL)[1], 1]),
                              K_EPI=np.asarray(
                                  self.et_indices[
                                  self.get_surface_vertex_start_end_index(
                                      Surface.LV_ENDOCARDIAL)[0]:
                                  self.get_surface_vertex_start_end_index(
                                      Surface.LV_ENDOCARDIAL)[1], 2]),
                                x=x, y=y, z=z)
        LVvol = 0
        for i in LvSurfaces:
            seStart = self.get_surface_start_end_index(surface_name=i)[0]
            #index where surface i start
            seEnd = self.get_surface_start_end_index(surface_name=i)[1]
            #index where surface i ends
            for se in range(seStart,seEnd+1):
                #range of surface triangles that comprise surface i
                indices = self.et_indices[se]
                #each se in range start-end corresponds to a set of 3 indices in et_pos
                Pts = self.et_pos[indices]
                #each of these 3 indices corresponds to a point, defined by x,y,z co-ords. Pts is 3x3 matrix of those points, and forms surface triangle.
                LVvol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D) #these three triangle points and centroid D form tetrahedron, volume of which is calculated
        LVvol /= 1000 #by iterating through all surface
        # triangles that comprise LV and adding the volumes,
        # you calculate the volume of the closed ventricle
        RVvol = 0
        for i in RvSurfaces: #same concept as for LV, different surfaces
            seStart = self.get_surface_start_end_index(surface_name=i)[0]
            seEnd = self.get_surface_start_end_index(surface_name=i)[1]
            for se in range(seStart,seEnd+1):
                indices = self.et_indices[se]
                Pts = self.et_pos[indices]

                RVvol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        RVvol /= 1000

        RVsVol = 0 #RV septum has inverted normal; volume of RVS must be
        # subtracted from RV_vol,
        #  rather than added like other surfaces
        seStart = self.get_surface_start_end_index(
            surface_name=Surface.RV_SEPTUM)[0]
        seEnd = self.get_surface_start_end_index(
            surface_name=Surface.RV_SEPTUM)[1]
        for se in range(seStart, seEnd + 1):
            indices = self.et_indices[se]
            Pts = self.et_pos[indices]
            RVsVol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        RVsVol /= 1000

        return(LVvol, RVvol-RVsVol) #RVS subtracted from RV to give proper RV_vol,
        # LV has no inverted normals. Both return in ml3

    def get_myocardial_mass(self, LVvol, RVvol):
        '''Calculates volume of closed epicardial surface, subtracts
        ventricular volumes and multiplies myocardium density to return in
        grams

        Parameters:
        ------------

        `RVvol` float RV ventricular volume

        `LVvol` float LV ventricular volume

        Returns:
        --------

        float myocardial mass
        '''

        D = [0,0,0] #arbitrary centroid point
        LVMyoVol = 0
        RVMyoVol = 0
        #LV Epicardium
        for se in range(2416,len(self.et_indices_EpiLVRV)):
            #same style as volume calcs, get volume of LV epicardium defined by EpiLVRV
            indices = self.et_indices_EpiLVRV[se]
            Pts = self.et_pos[indices]
            LVMyoVol += self.Get_tetrahedron_vol_CM(Pts[0],Pts[1],Pts[2],D)
        Lv_MyoVol_sum = LVMyoVol/1000

        #RV Epicardium
        for se in range(0,2416):
            indices = self.et_indices_EpiLVRV[se]
            Pts = self.et_pos[indices]
            RVMyoVol += self.Get_tetrahedron_vol_CM(Pts[0],Pts[1],Pts[2],D)
        RvMyoVol_sum = RVMyoVol/1000

        #ThruWall surface
        VolThru = 0 #thruwall surface divides LV and RV through septum
        # (which doesn't cover epicardium to close surfaces separately)
        for se in range(0,len(self.et_indices_thruWall)):
            indices = self.et_indices_thruWall[se]
            Pts = self.et_pos[indices]
            VolThru += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        Lv_MyoVol_sum -= VolThru/1000 #ThruWall normals inverted for LV, normal
        # for RV. Therefore subtract vol from LV, add to RV
        RvMyoVol_sum += VolThru/1000

        #Mitral/aortic valves
        vol = 0 #valve points for LV
        for i in [Surface.MITRAL_VALVE,Surface.AORTA_VALVE]:
            seStart = self.get_surface_start_end_index(surface_name=i)[0]
            seEnd = self.get_surface_start_end_index(surface_name=i)[1]
            for se in range(seStart, seEnd + 1):
                indices = self.et_indices[se]
                Pts = self.et_pos[indices]
                vol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        Lv_MyoVol_sum += vol/1000

        #Pulmonary/tricuspid valves
        vol = 0 #valve points for RV
        for i in [Surface.PULMONARY_VALVE,Surface.TRICUSPID_VALVE]:
            seStart = self.get_surface_start_end_index(surface_name=i)[0]
            seEnd = self.get_surface_start_end_index(surface_name=i)[1]
            for se in range(seStart, seEnd + 1):
                indices = self.et_indices[se]
                Pts = self.et_pos[indices]
                vol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        RvMyoVol_sum += vol/1000

        #Septum
        vol = 0
        seStart= self.get_surface_start_end_index(Surface.RV_SEPTUM)[0]
        seEnd = self.get_surface_start_end_index(Surface.RV_SEPTUM)[1]
        for se in range(seStart, seEnd + 1):
            indices = self.et_indices[se]
            Pts = self.et_pos[indices]
            vol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        Lv_MyoVol_sum += vol/1000 #Septum normals face LV, inverted for RV,
        # therefore vol added to LV and subtracted from RV
        RvMyoVol_sum -= vol/1000

        #Mass calculations
        LVmass = (Lv_MyoVol_sum - LVvol) * 1.05
        #1.05 is density of myocardial mass
        RVmass = (RvMyoVol_sum - RVvol) * 1.05
        #calculation is volume of each
        # ventricle epicardium-ventricular volume, * myocardium density
        return (LVmass, RVmass) #returns tuple of LV_myo_mass,
        # RV_myo_mass, both in grams

    def update_control_mesh(self,new_control_mesh):
        '''Update control mesh

        Parameters
        ----------

        new_control_mesh: (388,3) array of new control node positions

        '''
        self.control_mesh = new_control_mesh
        self.et_pos = np.linalg.multi_dot(
            [self.matrix, self.control_mesh])





