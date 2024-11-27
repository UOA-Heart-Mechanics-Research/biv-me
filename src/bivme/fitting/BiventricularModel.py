from scipy import spatial
import functools

import scipy
from copy import deepcopy

# local imports
from .GPDataSet import *
from .surface_enum import Surface
from .surface_enum import ContourType
from .surface_enum import SURFACE_CONTOUR_MAP
from .fitting_tools import *
from .build_model_tools import *

from collections import OrderedDict
from nltk import flatten
from loguru import logger
import pyvista as pv
import scipy.io

##Author : Charlène Mauger, University of Auckland, c.mauger@auckland.ac.nz
class BiventricularModel:
    """This class creates a surface from the control mesh, based on
    Catmull-Clark subdivision surface method. Surfaces have the following properties:

    Attributes:
       num_nodes = 388                       Number of control nodes.
       NUM_ELEMENTS = 187                    Number of elements.
       num_surface_nodes = 5810               Number of nodes after subdivision
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
       et_indices_control_mesh              Element connectivities (n1, n2, n3) for each face of the coarse mesh (708x3)
       basis_matrix                         Matrix (5810x388) containing basis
                                            functions used to evaluate surface
                                            at surface point locations
       matrix                               Subdivision matrix (388x5810).


       gtstsg_x, gtstsg_y, gtstsg_z         Regularization/Smoothing matrices
                                            (388x388) along
                                            Xi1 (circumferential),
                                            Xi2 (longitudinal) and
                                            Xi3 (transmural) directions


       APEX_INDEX                           Vertex index of the apex

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

       mbder_dx, mbder_dy, mbder_dz         Matrices (5049x338) containing
                                            weights used to calculate gradients
                                            of the displacement field at Gauss
                                            point locations.

       Jac11, jac_12, jac_13                  Matrices (11968x388) containing
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

    NUM_NODES = 388
    """Class constant, Number of control nodes (388)."""
    NUM_ELEMENTS = 187
    """Class constant, number of elements (187)."""
    NUM_SURFACE_NODES = 5810
    """class constant, number of nodes after subdivision (5810)."""
    APEX_INDEX = 5485 #  # 50 endo #5485 #epi
    """class constant, vertex index defined as the apex point."""
    NUM_GAUSSIAN_POINTS = 5049
    """Number of gaussian points"""
    NUM_NODES_THRU_WALL = 160
    """Number of points defining the thru wall"""
    NUM_SUBDIVIDED_FACES = 11760
    """Number of faces after subdivision"""
    NUM_COARSE_FACES = 708
    """Number of faces before subdivision"""
    NUM_LOCAL_POINTS = 12509
    """Number of local points - used for patch estimation"""

    control_mesh_vertex_start_end = np.array(
        [
            [0, 104], # LV_ENDOCARDIAL
            [105, 210], # RV_ENDOCARDIAL
            [211, 353], # EPICARDIAL
        ]
    )

    et_vertex_start_end = np.array(
        [
            [0, 1499],
            [1500, 2164],
            [2165, 3223],
            [3224, 5581],
            [5582, 5630],
            [5631, 5655],
            [5656, 5696],
            [5697, 5729],
            [5730, 5809],
        ]
    )
    """Class constant, surface index limits for vertices `et_pos`. 
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
    """
    surface_start_end = np.array(
        [
            [0, 3071],
            [3072, 4479],
            [4480, 6751],
            [6752, 11615],
            [11616, 11663],
            [11664, 11687],
            [11688, 11727],
            [11728, 11759],
        ]
    )
    """Class constant,  surface index limits for embedded triangles `et_indices`.
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
    """

    control_mesh_start_end = np.array(
        [
            [0, 191],   # LV_ENDOCARDIAL = 0
            [192, 421], # RV_ENDOCARDIAL = 1
            [422, 707], # EPICARDIAL = 2
        ]
    )
    """Class constant,  control mesh index limits for embedded triangles `et_indices_control_mesh`.
    Surfaces are defined in the following order

            LV_ENDOCARDIAL = 0
            RV_ENDOCARDIAL = 1
            EPICARDIAL = 2

    To get control mesh end and start vertex index use
    get_control_mesh_start_end_index(surface_name)
    """

    def __init__(self, control_mesh_dir: os.PathLike, label: str = "default", build_mode: bool = False, collision_detection: bool = False):
        """Return a Surface object whose control mesh should be
        fitted to the dataset *DataSet*.

        control_mesh is always the same - this is the RVLV template. If
        the template is changed, one needs to regenerate all the matrices.
        The build_mode allows to load the data needed to interpolate a
        surface field. For fitting purposes set build_mode to False
        """
        self.build_mode = build_mode
        """
        False by default, true to evaluate surface points at xi local 
        coordinates
        """

        assert control_mesh_dir.exists(), \
            f"Cannot find {control_mesh_dir}!"

        self.label = label
        model_file = control_mesh_dir / "model.txt"
        assert model_file.exists(), \
            f"Missing {model_file}!"

        self.control_mesh = (
            pd.read_table(model_file, sep=r'\s+', header=None, engine="c")
        ).values

        """ `numNodes`X3 array[float] of x,y,z coordinates of control mesh.
        """

        subdivision_matrix_file = control_mesh_dir / "subdivision_matrix_sparse.mat"
        assert subdivision_matrix_file.exists(), \
            f"Missing {subdivision_matrix_file}!"

        self.matrix = scipy.io.loadmat(subdivision_matrix_file)['S'].toarray()
        """Subdivision matrix (`numNodes`x`numSurfaceNodes`).
        """

        self.et_pos = np.dot(self.matrix, self.control_mesh)
        """`numSurfaceNodes`x3 array[float] of x,y,z coordinates for each
                                            surface nodes.
        """

        et_index_file = control_mesh_dir / "ETIndicesSorted.txt"
        assert et_index_file.exists(), \
            f"Missing {et_index_file}!"

        self.et_indices = (
                              pd.read_table(et_index_file, sep=r'\s+', header=None, engine="c")
                          ).values.astype(int) - 1
        """ 11760x3 array[int] of elements connectivity (n1,n2,n3) for each face."""

        material_file = control_mesh_dir / 'ETIndicesMaterials.txt'
        assert material_file.exists(), \
            f"Missing {et_index_file}"
        self.material = np.loadtxt(material_file, dtype='str')

        self.collision_detection = collision_detection

        if collision_detection:
            self.reference_collision = set(self.detect_collision())

        et_index_thru_wall_file = control_mesh_dir / "thru_wall_et_indices.txt"
        assert et_index_thru_wall_file.exists(), \
            f"Missing {et_index_thru_wall_file} for myocardial mass calculation"

        et_index_file = control_mesh_dir / "ETIndices_control_mesh.txt"
        assert et_index_file.exists(), \
            f"Missing {et_index_file}!"

        self.et_indices_control_mesh = (
                              pd.read_table(et_index_file, sep=r'\s+', header=None, engine="c")
                          ).values.astype(int) - 1
        """ 11760x3 array[int] of elements connectivity (n1,n2,n3) for each face of the coarse_mesh."""

        self.et_indices_thru_wall = (
                                       pd.read_table(et_index_thru_wall_file, sep=r'\s+', header=None)
                                   ).values.astype(int) - 1

        et_index_epi_lvrv_file = control_mesh_dir / "ETIndicesEpiRVLV.txt"  # RB addition for MyoMass calc
        assert et_index_epi_lvrv_file.exists(), \
            f"Missing {et_index_epi_lvrv_file} for myocardial mass calculation"

        self.et_indices_epi_lvrv = (
                                      pd.read_table(
                                          et_index_epi_lvrv_file, sep=r'\s+', header=None, engine="c"
                                      )
                                  ).values.astype(int) - 1

        gtstsg_x_file = control_mesh_dir / "GTSTG_x_sparse.mat"
        assert gtstsg_x_file.exists(), \
            f"Missing {gtstsg_x_file}"
        self.gtstsg_x = scipy.io.loadmat(gtstsg_x_file)['S'].toarray()
        """`numNodes`x`numNodes` Regularization/Smoothing matrix along Xi1 (
        circumferential direction)        
        """

        gtstsg_y_file = control_mesh_dir / "GTSTG_y_sparse.mat"
        assert gtstsg_y_file.exists(), \
            f"Missing {gtstsg_y_file}"
        self.gtstsg_y = scipy.io.loadmat(gtstsg_y_file)['S'].toarray()
        """`numNodes`x`numNodes` Regularization/Smoothing matrix along
                                            Xi2 (longitudinal) direction"""

        gtstsg_z_file = control_mesh_dir / "GTSTG_z_sparse.mat"
        assert gtstsg_z_file.exists(), \
            f"Missing {gtstsg_z_file}"
        self.gtstsg_z = scipy.io.loadmat(gtstsg_z_file)['S'].toarray()
        """`numNodes`x`numNodes` Regularization/Smoothing matrix along
                                                    Xi3 (transmural) direction"""

        et_vertex_element_num_file = control_mesh_dir / "etVertexElementNum.txt"
        assert et_vertex_element_num_file.exists(), \
            f"Missing {et_vertex_element_num_file}"
        self.et_vertex_element_num = (
                                         pd.read_table(
                                             et_vertex_element_num_file, sep=r'\s+', header=None, engine="c"
                                         )
                                     ).values[:, 0].astype(int) - 1
        """`numSurfaceNodes`x1 array[int] Element num for each surface nodes.
        Used for surface evaluation 
        """

        mbder_x_file = control_mesh_dir / "mBder_x_sparse.mat"
        assert mbder_x_file.exists(), \
            f"Missing {mbder_x_file}"
        self.mbder_dx = scipy.io.loadmat(mbder_x_file)['S'].toarray()


        """`numSurfaceNodes`x`numNodes` Matrix containing  weights used to 
        calculate gradients of the displacement field at Gauss point locations.
        """

        mbder_y_file = control_mesh_dir / "mBder_y_sparse.mat"
        assert mbder_y_file.exists(), \
            f"Missing {mbder_y_file}"
        self.mbder_dy = scipy.io.loadmat(mbder_y_file)['S'].toarray()
        """`numSurfaceNodes`x`numNodes` Matrix containing  weights used to 
        calculate gradients of the displacement field at Gauss point locations.
        """

        mbder_z_file = control_mesh_dir / "mBder_z_sparse.mat"
        assert mbder_z_file.exists(), \
            f"Missing {mbder_z_file}"
        self.mbder_dz = scipy.io.loadmat(mbder_z_file)['S'].toarray()
        """`numSurfaceNodes`x`numNodes` Matrix containing  weights used to 
        calculate gradients of the displacement field at Gauss point locations.
        """

        jac_11_file = control_mesh_dir / "J11_sparse.mat"
        assert jac_11_file.exists(), \
            f"Missing {jac11_file}"
        self.jac_11 = scipy.io.loadmat(jac_11_file)['S'].toarray()
        """11968 x `numNodes` matrix containing weights used to calculate 
        Jacobians  along Xi1 at Gauss point location.
        Each matrix element is a linear combination of the 388 control points.
        """

        jac_12_file = control_mesh_dir / "J12_sparse.mat"
        assert jac_12_file.exists(), \
            f"Missing {jac_12_file}"
        self.jac_12 = scipy.io.loadmat(jac_12_file)['S'].toarray()
        """11968 x `numNodes` matrix containing weights used to calculate 
        Jacobians  along Xi2 at Gauss point location.
        Each matrix element is a linear combination of the 388 control points.
        """

        jac_13_file = control_mesh_dir / "J13_sparse.mat"
        assert jac_13_file.exists(), \
            f"Missing {jac_13_file}"
        self.jac_13 = scipy.io.loadmat(jac_13_file)['S'].toarray()
        """11968 x `numNodes` matrix containing weights used to calculate 
        Jacobians along Xi3 direction at Gauss point location.
        Each matrix element is a linear combination of the 388 control points.
        """

        basic_matrix_file = control_mesh_dir / "basis_matrix_sparse.mat"
        assert basic_matrix_file.exists(), \
            f"Missing {basic_matrix_file}"
        self.basis_matrix = scipy.io.loadmat(basic_matrix_file)['S'].toarray()

        """`numSurfaceNodes`x`numNodes` array[float]  basis  functions used 
        to evaluate surface at surface point locations
        """

        if not self.build_mode:
            return

        et_vertex_xi_file = control_mesh_dir / "etVertexXi.txt"
        assert et_vertex_xi_file.exists(), \
            f"Missing {et_vertex_xi_file}"
        self.et_vertex_xi = (
            pd.read_table(
                et_vertex_xi_file, sep=r'\s+', header=None, engine="c"
            )
        ).values
        """ `numSurfaceNodes`x3 array[float] of local xi position (xi1,xi2,
        xi3) for each vertex.
        """

        b_spline_file = control_mesh_dir / "control_points_patches.txt"
        assert b_spline_file.exists(), \
            f"Missing {b_spline_file}"
        self.b_spline = (
                            pd.read_table(b_spline_file, sep=r'\s+', header=None, engine="c")
                        ).values.astype(int) - 1
        """ numSurfaceNodesX32 array[int] of 32 control points which need to be 
         weighted (for each vertex)
        """

        boundary_file = control_mesh_dir / "boundary.txt"
        assert boundary_file.exists(), \
            f"Missing {boundary_file}"
        self.boundary = (
            pd.read_table(boundary_file, sep=r'\s+', header=None, engine="c")
        ).values.astype(int)
        """ boundary"""

        control_ef_file = control_mesh_dir / "control_mesh_connectivity.txt"
        assert control_ef_file.exists(), \
            f"Missing {control_ef_file}"
        self.control_et_indices = (
                                      pd.read_table(
                                          control_ef_file, sep=r'\s+', header=None, engine="c"
                                      )
                                  ).values.astype(int) - 1
        """ (K,8) matrix of control mesh connectivity"""

        phantom_points_file = control_mesh_dir / "phantom_points.txt"
        assert phantom_points_file.exists(), \
            f"Missing {phantom_points_file}"
        self.phantom_points = (
            pd.read_table(
                phantom_points_file, sep=r'\s+', header=None, engine="c"
            )
        ).values.astype(float)
        """ Some surface nodes are not needed for the 
        definition of the biventricular 2D surface therefore they are 
        not include in the surface node matrix. However they are 
        necessary for the 3D interpolation (septum area).
        these elements are called the phantom points and the 
        corresponding information as the subdivision level , local
        patch coordinates etc. are stored in phantom points array
        """

        self.phantom_points[:, :17] = self.phantom_points[:, :17].astype(int) - 1
        patch_coordinates_file = control_mesh_dir / "patch_coordinates.txt"
        assert patch_coordinates_file.exists(), \
            f"Missing {patch_coordinates_file}"
        self.patch_coordinates = (
            pd.read_table(
                patch_coordinates_file, sep=r'\s+', header=None, engine="c"
            )
        ).values
        """local patch coordinates. 

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
        """
        fraction_file = control_mesh_dir / "fraction.txt"
        assert fraction_file.exists(), \
            f"Missing {fraction_file}"
        self.fraction = (
            pd.read_table(fraction_file, sep=r'\s+', header=None, engine="c")
        ).values
        """`numSurfaceNodes`x1 vector[int] subdivision level of the 
         patch (level 0 = 1,level 1 = 0.5,level 2 = 0.25). See 
         `patch_coordinates for details`
        """

        local_matrix_file = control_mesh_dir / "local_matrix_sparse.mat"
        assert local_matrix_file.exists(), \
            f"Missing {local_matrix_file}"
        self.local_matrix = scipy.io.loadmat(local_matrix_file)['S'].toarray()

    def get_nodes(self) -> np.ndarray:
        """
        Returns
        --------
        `NUM_SURFACE_NODES`x3 array of vertices coordinates
        """
        return self.et_pos

    def get_control_mesh_nodes(self) -> np.ndarray:
        """
        Returns
        -------
        `NUM_NODES`x3 array of coordinates of control points
        """
        return self.control_mesh

    def get_surface_vertex_start_end_index(self, surface_name: Surface) -> np.ndarray:
        """Return first and last vertex index for a given surface to use
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
            return [self.APEX_INDEX] * 2

    def get_surface_faces(self, surface: Surface) -> np.ndarray:
        """Get the faces definition for a surface triangulation"""

        surface_index = self.get_surface_start_end_index(surface)
        return self.et_indices[surface_index[0] : surface_index[1] + 1, :]

    def get_surface_start_end_index(self, surface_name: Surface) -> np.ndarray:
        """Return first and last element index for a given surface, tu use
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



























    def get_control_mesh_vertex_start_end_index(self, surface_name: ControlMesh) -> np.ndarray:
        """Return first and last vertex index for a given surface to use
        with `et_pos` array

        Parameters
        -----------

        `surface_name`  Surface name as defined in 'Surface' enumeration

        `Returns`
        ---------
        2x1 array with first and last vertices index belonging to
            surface_name
        """

        if surface_name == ControlMesh.LV_ENDOCARDIAL:
            return self.control_mesh_vertex_start_end[0, :]

        if surface_name == ControlMesh.RV_ENDOCARDIAL:
            return self.control_mesh_vertex_start_end[1, :]

        if surface_name == Surface.EPICARDIAL:
            return self.control_mesh_vertex_start_end[2, :]

        #if surface_name == Surface.MITRAL_VALVE:
        #    return self.et_vertex_start_end[4, :]
#
        #if surface_name == Surface.AORTA_VALVE:
        #    return self.et_vertex_start_end[5, :]
#
        #if surface_name == Surface.TRICUSPID_VALVE:
        #    return self.et_vertex_start_end[6, :]
#
        #if surface_name == Surface.PULMONARY_VALVE:
        #    return self.et_vertex_start_end[7, :]

    def get_control_mesh_faces(self, surface: ControlMesh) -> np.ndarray:
        """Get the faces definition for a surface triangulation"""

        surface_index = self.get_control_mesh_start_end_index(surface)
        return self.et_indices_control_mesh[surface_index[0] : surface_index[1] + 1, :]

    def get_control_mesh_start_end_index(self, surface_name: ControlMesh) -> np.ndarray:
        """Return first and last element index for a given surface, tu use
        with `et_indices_control_mesh` array
        Parameters
        ----------
        `surface_name` surface name as defined by `ControlMesh` enum

        Returns
        -------
        2x1 array containing first and last vertices index belonging to
           `surface_name`
        """

        if surface_name == ControlMesh.LV_ENDOCARDIAL:
            return self.surface_start_end[0, :]

        if surface_name == ControlMesh.RV_ENDOCARDIAL:
            return self.surface_start_end[1, :]

        if surface_name == ControlMesh.EPICARDIAL:
            return self.surface_start_end[2, :]

        #if surface_name == ControlMesh.MITRAL_VALVE:
        #    return self.surface_start_end[4, :]
#
        #if surface_name == ControlMesh.AORTA_VALVE:
        #    return self.surface_start_end[5, :]
#
        #if surface_name == ControlMesh.TRICUSPID_VALVE:
        #    return self.surface_start_end[6, :]
#
        #if surface_name == ControlMesh.PULMONARY_VALVE:
        #    return self.surface_start_end[7, :]

    def is_diffeomorphic(self, updated_control_mesh: np.ndarray, min_jacobian: float) -> bool:
        """This function checks the Jacobian value at Gauss point location
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
            boolean value
        """

        for i in range(len(self.jac_11)):
            jacobi = np.array(
                [
                    [
                        np.inner(self.jac_11[i, :], updated_control_mesh[:, 0]),
                        np.inner(self.jac_12[i, :], updated_control_mesh[:, 0]),
                        np.inner(self.jac_13[i, :], updated_control_mesh[:, 0]),
                    ],
                    [
                        np.inner(self.jac_11[i, :], updated_control_mesh[:, 1]),
                        np.inner(self.jac_12[i, :], updated_control_mesh[:, 1]),
                        np.inner(self.jac_13[i, :], updated_control_mesh[:, 1]),
                    ],
                    [
                        np.inner(self.jac_11[i, :], updated_control_mesh[:, 2]),
                        np.inner(self.jac_12[i, :], updated_control_mesh[:, 2]),
                        np.inner(self.jac_13[i, :], updated_control_mesh[:, 2]),
                    ],
                ]
            )

            determinant = np.linalg.det(jacobi)
            if determinant < min_jacobian:
                return False

        return True

    def update_pose_and_scale(self, dataset: GPDataSet) -> None:
        """A method that scale and translate the model to rigidly align
        with the guide points.

        Notes
        ------
        Parameters
        ------------
        `dataset` GPDataSet object with guide points
        Returns
        --------

        `scale_factor` scale factor between template and data points.
        """

        scale_factor = self.get_scaling(dataset)
        self.update_control_mesh(self.control_mesh * scale_factor)

        # The rotation is defined about the origin so we need to translate the model to the origin
        self.update_control_mesh(self.control_mesh - self.et_pos.mean(axis=0))
        rotation = self.get_rotation(dataset)
        self.update_control_mesh(
            np.array([np.dot(rotation, node) for node in self.control_mesh])
        )

        # Translate the model back to origin of the DataSet coordinate system
        translation = self.get_translation(dataset)

        self.update_control_mesh(self.control_mesh + translation)

    def get_scaling(self, dataset: GPDataSet) -> float:
        """Calculates a scaling factor for the model
        to match the guide points defined in dataset

        Parameters
        -----------
        `data_set` GPDataSet object

        Returns
        --------
        `scaleFactor` float
        """
        model_shape_index = [
            self.APEX_INDEX,
            self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1],
            self.get_surface_vertex_start_end_index(Surface.TRICUSPID_VALVE)[1],
        ]
        model_shape = np.array(self.et_pos[model_shape_index, :])
        reference_shape = np.array(
            [dataset.apex, dataset.mitral_centroid, dataset.tricuspid_centroid]
        )
        mean_m = model_shape.mean(axis=0)
        mean_r = reference_shape.mean(axis=0)
        model_shape = model_shape - mean_m
        reference_shape = reference_shape - mean_r
        ss_model = (model_shape**2).sum()
        ss_reference = (reference_shape**2).sum()

        # centered Forbidius norm
        norm_model = np.sqrt(ss_model)
        reference_norm = np.sqrt(ss_reference)

        scale_factor = reference_norm / norm_model

        return scale_factor

    def get_translation(self, dataset: GPDataSet) -> np.ndarray:
        """Calculates a translation for (x, y, z)
        axis that aligns the model RV center with dataset RV center
        Parameters
        -----------
        `data_set` GPDataSet object

        Returns
        --------
          `translation` 3X1 array[float] with x, y and z translation
        """

        dataset_coordinates = [dataset.apex, dataset.mitral_centroid, dataset.tricuspid_centroid]
        model_point_indices = [self.APEX_INDEX, self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1], self.get_surface_vertex_start_end_index(Surface.TRICUSPID_VALVE)[1]]
        model_coordinates = self.et_pos[model_point_indices,:]
        translation = np.mean(dataset_coordinates, axis=0) - np.mean(model_coordinates,axis=0)

        return translation

    def get_rotation(self, data_set: GPDataSet) -> np.ndarray:
        """Computes the rotation between model and data set,
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
        """

        base = data_set.mitral_centroid
        base_model = self.et_pos[
                     self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1], :
                     ]

        # computes data_set coordinates system
        x_axis = data_set.apex - base
        x_axis = x_axis / np.linalg.norm(x_axis)

        apex_position_model = self.et_pos[self.APEX_INDEX, :]

        x_axis_model = apex_position_model - base_model
        x_axis_model = x_axis_model / np.linalg.norm(x_axis_model)  # normalize

        # compute origin defined at 1/3 of the height of the model on the Ox
        # axis
        temp_original = 0.5 * (data_set.apex + base)
        temp_original_model = 0.5 * (apex_position_model + base_model)

        max_d = np.linalg.norm(0.5 * (data_set.apex - base))
        min_d = -np.linalg.norm(0.5 * (data_set.apex - base))

        max_d_model = np.linalg.norm(0.5 * (apex_position_model - base_model))
        min_d_model = -np.linalg.norm(0.5 * (apex_position_model - base_model))

        point_proj = data_set.points_coordinates[
            (data_set.contour_type == ContourType.SAX_LV_ENDOCARDIAL), :
        ]

        point_proj = np.vstack(
            (
                point_proj,
                data_set.points_coordinates[
                    (data_set.contour_type == ContourType.LAX_LV_ENDOCARDIAL), :
                ]
            )
        )

        assert len(point_proj) > 0, \
            f"No LV contours found in get_rotation"

        temp_d = [np.dot(x_axis, p) for p in (point_proj - temp_original)]
        max_d = max(np.max(temp_d), max_d)
        min_d = min(np.min(temp_d), min_d)

        point_proj_model = self.et_pos[
            self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[
                0
            ] : self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[1]
            + 1,
            :,
        ]

        temp_d_model = [
            np.dot(x_axis_model, point_model)
            for point_model in (point_proj_model - temp_original_model)
        ]
        max_d_model = max(np.max(temp_d_model), max_d_model)
        min_d_model = min(np.min(temp_d_model), min_d_model)

        centroid = temp_original + min_d * x_axis + ((max_d - min_d) / 3.0) * x_axis
        centroid_model = (
            temp_original_model
            + min_d_model * x_axis_model
            + ((max_d_model - min_d_model) / 3.0) * x_axis_model
        )

        # Compute Oy axis
        valid_index = (data_set.contour_type == ContourType.SAX_RV_FREEWALL) + (
            data_set.contour_type == ContourType.SAX_RV_SEPTUM
        ) + (data_set.contour_type == ContourType.LAX_RV_FREEWALL) + (
            data_set.contour_type == ContourType.LAX_RV_SEPTUM
        )

        rv_endo_points = data_set.points_coordinates[valid_index, :]

        rv_endo_points_model = self.et_pos[
            self.get_surface_vertex_start_end_index(Surface.RV_SEPTUM)[
                0
            ]:self.get_surface_vertex_start_end_index(Surface.RV_FREEWALL)[1]
            + 1,
            :,
        ]

        rv_centroid = rv_endo_points.mean(axis=0)
        rv_centroid_model = rv_endo_points_model.mean(axis=0)

        scale = np.dot(x_axis, rv_centroid) - np.dot(x_axis, centroid) / np.dot(
            x_axis, x_axis
        )
        scale_model = np.dot(x_axis_model, rv_centroid_model) - np.dot(
            x_axis_model, centroid_model
        ) / np.dot(x_axis_model, x_axis_model)
        rv_proj = centroid + scale * x_axis
        rv_proj_model = centroid_model + scale_model * x_axis_model

        y_axis = rv_centroid - rv_proj
        y_axis_model = rv_centroid_model - rv_proj_model

        y_axis /= np.linalg.norm(y_axis)
        y_axis_model /= np.linalg.norm(y_axis_model)

        z_axis = np.cross(x_axis, y_axis)
        z_axis_model = np.cross(x_axis_model, y_axis_model)

        # normalization
        z_axis /= np.linalg.norm(z_axis)
        z_axis_model /= np.linalg.norm(z_axis_model)

        # Find translation and rotation between the two coordinates systems
        # The easiest way to solve it (in my opinion) is by using a
        # Singular Value Decomposition as reported by Markley (1988):
        #    1. Obtain a matrix B as follows:
        #        B=∑ni=1aiwiviTB=∑i=wiviT
        #    2. Find the SVD of BB
        #        B=USVT
        #    3. The rotation matrix is:
        #        R=UMVT, where M=diag([11det(U)det(V)])

        # Step 1
        b = (
            np.outer(x_axis, x_axis_model)
            + np.outer(y_axis, y_axis_model)
            + np.outer(z_axis, z_axis_model)
        )

        # Step 2
        [u, _, v_t] = np.linalg.svd(b)

        m = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(u) * np.linalg.det(v_t)]]
        )
        rotation = np.dot(u, np.dot(m, v_t))

        return rotation

    def update_control_mesh(self, new_control_mesh: np.ndarray) -> None:
        """Update control mesh
        Parameters
        ----------
        new_control_mesh: (388,3) array of new control node positions
        """
        self.control_mesh = new_control_mesh
        self.et_pos = np.linalg.multi_dot([self.matrix, self.control_mesh])

    def detect_collision(self, debug: bool = False) -> list:
        ##TODO Initialise pv meshes is collision detection set to on

        from bivme.meshing.mesh import Mesh
        model = Mesh('mesh')
        model.set_nodes(self.et_pos)
        model.set_elements(self.et_indices)

        ## convert labels to integer corresponding to the sorted list of unique labels types
        unique_material = np.unique(self.material[:,1])

        materials = np.zeros(self.material.shape)
        for index, m in enumerate(unique_material):
            face_index = self.material[:, 1] == m
            materials[face_index, 0] = self.material[face_index, 0].astype(int)
            materials[face_index, 1] = [index] * np.sum(face_index)

        model.set_materials(materials[:, 0], materials[:, 1])
        # components list, used to get the correct mesh components:
        # ['0 AORTA_VALVE' '1 AORTA_VALVE_CUT' '2 LV_ENDOCARDIAL' '3 LV_EPICARDIAL'
        # ' 4 MITRAL_VALVE' '5 MITRAL_VALVE_CUT' '6 PULMONARY_VALVE' '7 PULMONARY_VALVE_CUT'
        # '8 RV_EPICARDIAL' '9 RV_FREEWALL' '10 RV_SEPTUM' '11 TRICUSPID_VALVE'
        # '12 TRICUSPID_VALVE_CUT', '13' THRU WALL]

        rv_fw = model.get_mesh_component([9], reindex_nodes=False)
        rv_septum = model.get_mesh_component([10], reindex_nodes=False)

        rv_fw_faces = rv_fw.elements
        rv_fw_et = np.pad(rv_fw_faces, ((0, 0), (1, 0)), 'constant', constant_values=3)
        rv_septum_et = np.pad(rv_septum.elements, ((0, 0), (1, 0)), 'constant', constant_values=3)       

        rvfw_mesh = pv.PolyData(rv_fw.nodes, rv_fw_et)
        rvs_mesh = pv.PolyData(rv_septum.nodes, rv_septum_et)

        collision, n_contacts = rvs_mesh.collision(rvfw_mesh, contact_mode=0, cell_tolerance=0)  

        scalars = np.zeros(collision.n_cells, dtype=bool)
        scalars[collision.field_data['ContactCells']] = True

        if debug:
            pl = pv.Plotter()
            _ = pl.add_mesh(
                collision,
                scalars=scalars,
                show_scalar_bar=False,
                cmap='bwr',)

            _ = pl.add_mesh(
                rvfw_mesh,
                style='wireframe',
                color='k',
                show_edges=True,)
            pl.show()

        return set(collision.field_data['ContactCells'])

    def plot_surface(
        self, face_color_lv: str="rgb(0,127,0)", face_color_rv : str="rgb(0,127,127)", face_color_epi : str="rgb(127,0,0)", surface: str="all"
    ) -> list:
        """Plot 3D model.

        Parameters
        -----------

        `face_color_lv` LV_ENDOCARDIAL surface color
        `face_color_rv` RV surface color
        `face_color_epi` Epicardial color
        `surface` surface to plot, default all = entire surface,
                    endo = endocardium, epi = epicardium
        Returns
        --------
        `triangles_epi` Nx3 array[int] triangles defining the epicardium surface
        `triangles_lv` Nx3 array[int] triangles defining the LV endocardium
        surface
        `triangles_RV` Nx3 array[int]  triangles defining the RV surface
        `lines` lines that need to be plotted
        """

        if surface not in ["all", "endo", "epi"] :
            logger.warning(f"Invalid surface argument in BiventricularModel::plot_surface. Got {surface}. Possible values (all, endo, epi). The model will not show on the htlm file")
            return []

        x = np.array(self.et_pos[:, 0]).T
        y = np.array(self.et_pos[:, 1]).T
        z = np.array(self.et_pos[:, 2]).T

        surface_enum = []
        if surface == "endo" or surface == "all":
            surface_enum.append({
                "surface": Surface.LV_ENDOCARDIAL,
                "color": face_color_lv,
                "name": "LV endocardium",
                "opacity": 1
            })
            surface_enum.append({
                "surface": Surface.RV_FREEWALL,
                "color": face_color_rv,
                "name": "RV free wall",
                "opacity": 1
            })
            surface_enum.append({
                "surface": Surface.RV_SEPTUM,
                "color": face_color_rv,
                "name": "RV septum",
                "opacity": 1
            })

        if surface == "all":
            surface_enum.append({
                "surface": Surface.EPICARDIAL,
                "color": face_color_epi,
                "name": "Epicardium",
                "opacity": 0.4
            })

        if surface == "epi": # we do not append here
            surface_enum = [{
                "surface": Surface.EPICARDIAL,
                "color": face_color_epi,
                "name": "Epicardium",
                "opacity": 1
            }]

        output = []

        for surface_type in surface_enum:
            surface_index = self.get_surface_start_end_index(surface_type["surface"])

            ijk = [np.asarray(self.et_indices[surface_index[0]: surface_index[1] + 1, idx]) for idx in range(0,3)]

            simplices = [self.et_indices[surface_index[0] : surface_index[1] + 1, idx] for idx in range(0,3)]

            points_3d = np.vstack(
                (self.et_pos[:, 0], self.et_pos[:, 1], self.et_pos[:, 2])
            ).T

            tri_vertices = list(map(lambda index: points_3d[index], np.asarray(simplices).T))
            triangles = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=surface_type["color"],
                i=ijk[0],
                j=ijk[1],
                k=ijk[2],
                opacity=surface_type["opacity"],
                name=surface_type["name"],
                showlegend=True,
            )

            output.append(triangles)

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices]
                for c in range(3)
            ]
            xe, ye, ze = [
                functools.reduce(lambda x, y: x + y, lists_coord[k]) for k in range(3)
            ]

            # define the lines to be plotted
            lines = go.Scatter3d(
                x=xe,
                y=ye,
                z=ze,
                mode="lines",
                line=go.scatter3d.Line(color="rgb(0,0,0)", width=1.5),
                showlegend=True,
                name=f"wireframe {surface_type['name']}",
            )

            output.append(lines)

        return output

    def get_intersection_with_plane(self, po: np.ndarray, no: np.ndarray, surface_to_use: Surface=None) -> np.ndarray:
        """Calculate intersection points between a plane with the
        biventricular model (LV_ENDOCARDIAL only)

        Parameters
        ----------
        `po` (3,1) array[float] a point of the plane
        `no` (3,1) array[float normal to the plane

        Returns
        -------

        `f_idx` (N,3) array[float] are indices of the surface nodes indicating
        intersecting the plane"""

        # Adjust po & no into a column vector
        no = no / np.linalg.norm(no)

        f_idx = []

        if surface_to_use is None:
            surface_to_use = [Surface.LV_ENDOCARDIAL]
        for surface in surface_to_use:  # We just want intersection LV_ENDOCARDIAL,
            # RVS. RVFW, epi
            # Get the faces
            faces = self.get_surface_faces(surface)

            # --- find triangles that intersect with plane: (po,no)
            # calculate sign distance of each vertices

            # set the origin of the model at po
            centered_vertex = self.et_pos - [list(po)] * len(self.et_pos)
            # projection on the normal
            dist = np.dot(no, centered_vertex.T)

            signed_distance = np.sign(dist[faces])
            # search for triangles having the vertex on the both sides of the
            # slice plane => intersecting with the slice plane
            valid_index = [
                np.any(signed_distance[i] > 0) and np.any(signed_distance[i] < 0)
                for i in range(len(signed_distance))
            ]
            intersecting_face_idx = np.where(valid_index)[0]

            if len(intersecting_face_idx) < 0:
                return np.empty((0, 3))

            # Find the intersection lines - find segments for each intersected
            # triangles that intersects the plane
            # see http://softsurfer.com/Archive/algorithm_0104/algorithm_0104B.htm

            # pivot points

            i_pos = [
                x for x in intersecting_face_idx if np.sum(signed_distance[x] > 0) == 1
            ]  # all
            # triangles with one vertex on the positive part
            i_neg = [
                x for x in intersecting_face_idx if np.sum(signed_distance[x] < 0) == 1
            ]  # all
            # triangles with one vertex on the negative part
            p1 = []
            u = []

            for face_index in i_pos:  # triangles where only one
                # point
                # on positive side
                # pivot points
                pivot_point_mask = signed_distance[face_index, :] > 0
                res = centered_vertex[faces[face_index, pivot_point_mask], :][0]
                p1.append(list(res))
                # u vectors
                u = u + list(
                    np.subtract(
                        centered_vertex[
                            faces[face_index, np.invert(pivot_point_mask)], :
                        ],
                        [list(res)] * 2,
                    )
                )

            for face_index in i_neg:  # triangles where only one
                # point on negative side
                # pivot points
                pivot_point_mask = signed_distance[face_index, :] < 0
                res = centered_vertex[faces[face_index, pivot_point_mask], :][
                    0
                ]  # select the vertex on the negative side
                p1.append(res)
                # u vectors
                u = u + list(
                    np.subtract(
                        centered_vertex[
                            faces[face_index, np.invert(pivot_point_mask)], :
                        ],
                        [list(res)] * 2,
                    )
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

            si = -np.dot(no.T, p1) / (np.dot(no.T, u))
            factor_u = np.array([list(si)] * 3)
            pts = np.add(p1, np.multiply(factor_u, u)).T
            # add vertices that are on the surface
            pon = centered_vertex[faces[signed_distance == 0], :]
            pts = np.vstack((pts, pon))
            # #change points to the original position
            f_idx = f_idx + list(pts + [list(po)] * len(pts))

        return f_idx

    def get_intersection_with_dicom_image(self, slice: Slice, surface: Surface=None) -> np.ndarray:
        """Get the intersection contour points between the biventricular
        model with a DICOM image

        Example
        -------
            obj.get_intersection_with_dicom_image(slice, Surface.RV_SEPTUM)

        Parameters
        ----------

        `slice` Slice obj with the dicom information

        `surface` Surface enum, model surface to be intersected

        Returns
        -------

        `P` (n,3) array[float] intersecting points
        """

        image_position = np.asarray(slice.position, dtype=float)
        image_orientation = np.asarray(slice.orientation, dtype=float)

        # get image position and the image vectors
        v1 = np.asarray(image_orientation[0:3], dtype=float)
        v2 = np.asarray(image_orientation[3:6], dtype=float)
        v3 = np.cross(v1, v2)

        # get intersection points
        p = self.get_intersection_with_plane(image_position, v3, surface_to_use=surface)

        return p

    def compute_data_xi(self, weight: float, data: GPDataSet) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        `psi_matrix` basis function matrix (`N`,`NUM_NODES`)

        """

        data_points = np.array(data.points_coordinates)
        data_contour_type = np.array(data.contour_type)
        data_weights = np.array(data.weights)
        psi_matrix = []
        w = []
        distance_d_prior = []
        index = []
        data_points_index = []
        index_unique = []  # add by LDT 3/11/2021

        basis_matrix = self.basis_matrix

        # add by A. Mira : a more compressed way of initializing the cKDTree

        for surface in Surface:
            # Trees initialization
            surface_index = self.get_surface_vertex_start_end_index(surface)
            tree_points = self.et_pos[surface_index[0] : surface_index[1] + 1, :]
            if len(tree_points) == 0:
                continue
            surface_tree = scipy.spatial.cKDTree(tree_points)

            # loop over contours is faster, for the same contours we are using
            # the same tree, therefore the query operation can be done for all
            # points of the same contour: A. Mira 02/2020
            for contour in SURFACE_CONTOUR_MAP[surface.value]:
                contour_points_index = np.where(data_contour_type == contour)[0]
                contour_points = data_points[contour_points_index]

                # if np.isnan(np.sum(contour_points))==True:
                # LDT 7/03: handle error, why do I get nan in contours?
                # continue

                weights_gp = data_weights[contour_points_index]

                if len(contour_points) == 0:
                    continue

                if surface.value < 4:  # these are the surfaces
                    distance, vertex_index = surface_tree.query(
                        contour_points, k=1, p=2
                    )
                    index_closest = [x + surface_index[0] for x in vertex_index]
                    # add by LDT (3/11/2021): perform preliminary operations for vertex points that are not in index
                    # instead of doing them in the 'else' below. This makes the for loop below faster.
                    unique_index_closest = list(
                        OrderedDict.fromkeys(index_closest)
                    )  # creates a list of elements that are unique in index_closest
                    dict_unique = dict(
                        zip(unique_index_closest, range(0, len(unique_index_closest)))
                    )  # create a dictionary = {'unique element': its list index}
                    vertex = list(dict_unique.keys())  # list of all the dictionary keys
                    common_elm = list(
                        set(index_unique) & set(vertex)
                    )  # intersection between the array index_unique and the unique points in index_closest

                    def filter_new(full_list, excludes):
                        """
                        eliminates the items in 'exclude' out of the full_list
                        """
                        s = set(excludes)
                        return (x for x in full_list if x not in s)

                    # togli gli elementi comuni
                    index_unique.append(
                        list(filter_new(vertex, common_elm))
                    )  # stores the new vertices that are NOT in already in the index_unique list
                    index_unique = flatten(index_unique)

                    items_as_dict = dict(
                        zip(index_unique, range(0, len(index_unique)))
                    )  # builds a dictionary = {vertices: indexes}

                    for i_idx, vertex_index in enumerate(index_closest):
                        if (
                            len(set([vertex_index]).intersection(index)) == 0
                        ):  # changed by LDT (3/11/2021): faster
                            index.append(int(vertex_index))
                            data_points_index.append(contour_points_index[i_idx])
                            psi_matrix.append(basis_matrix[int(vertex_index), :])
                            w.append(weight * weights_gp[i_idx])
                            distance_d_prior.append(distance[i_idx])

                        else:
                            old_idx = items_as_dict[
                                vertex_index
                            ]  # changed by LDT (3/11/2021)
                            distance_old = distance_d_prior[old_idx]
                            if distance[i_idx] < distance_old:
                                distance_d_prior[old_idx] = distance[i_idx]
                                data_points_index[old_idx] = contour_points_index[i_idx]
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
                        centroid_gp_valve = contour_points.mean(axis=0)
                        translation_gp_model = centroid_valve - centroid_gp_valve
                        translated_points = np.add(contour_points, translation_gp_model)

                    else:  # rv_inserts  and apex don't
                        # need to be translated
                        translated_points = contour_points

                    if contour in [
                        ContourType.MITRAL_PHANTOM,
                        ContourType.PULMONARY_PHANTOM,
                        ContourType.AORTA_PHANTOM,
                        ContourType.TRICUSPID_PHANTOM,
                    ]:
                        surface_tree = scipy.spatial.cKDTree(translated_points)
                        tree_points = tree_points[:-1]
                        distance, vertex_index = surface_tree.query(
                            tree_points, k=1, p=2
                        )
                        index_closest = [
                            x + surface_index[0] for x in range(len(tree_points))
                        ]
                        weights_gp = [weights_gp[x] for x in vertex_index]

                        contour_points_index = [
                            contour_points_index[x] for x in vertex_index
                        ]

                    else:
                        distance, vertex_index = surface_tree.query(
                            translated_points, k=1, p=2
                        )
                        index_closest = []
                        for x in vertex_index:
                            if (x + surface_index[0]) != surface_index[1]:
                                index_closest.append(x + surface_index[0])
                            else:
                                index_closest.append(x + surface_index[0] - 1)

                    index = index + index_closest
                    psi_matrix = psi_matrix + list(basis_matrix[index_closest, :])

                    w = w + [(weight * x) for x in weights_gp]
                    distance_d_prior = distance_d_prior + list(distance)
                    data_points_index = data_points_index + list(contour_points_index)

        return [
            np.asarray(data_points_index),
            np.asarray(w),
            np.asarray(distance_d_prior),
            np.asarray(psi_matrix),
        ]

    def extract_linear_hex_mesh(self, reorder_nodes : bool=True) -> tuple[np.ndarray, np.ndarray]:
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

        `new_nodes_position` (NUM_NODES,3) array[float] new position of the nodes
        `new_elements` (nbElem, 8) array mesh connectivity

        """

        new_elements = np.zeros_like(self.control_et_indices)
        if reorder_nodes:
            new_nodes_position = np.zeros_like(self.control_mesh)
            nodes_id = np.sort(np.unique(self.control_et_indices))
        else:
            new_nodes_position = np.zeros_like(self.et_pos)
        # node_maping = np.zeros(mesh.et_pos.shape[0])
        xi_order = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
        node_elem_map = self.et_vertex_element_num
        node_position = self.et_pos
        xi_position = self.et_vertex_xi

        for elem_id, elem_nodes in enumerate(self.control_et_indices):
            #
            elem_index = np.where(np.equal(node_elem_map, elem_id))[0]
            elem_index = elem_index.reshape(elem_index.shape[0])
            enode_position = node_position[elem_index]
            enode_xi = xi_position[elem_index, :]

            for node_index, node in enumerate(elem_nodes):
                index = np.prod(enode_xi == xi_order[node_index], axis=1).astype(bool)
                if index.any():
                    if reorder_nodes:
                        new_node_id = np.where(nodes_id == node)[0][0]
                    else:
                        new_node_id = node
                    new_nodes_position[new_node_id] = enode_position[index][0]
                    elem_index = self.control_et_indices == node
                    new_elements[elem_index] = new_node_id
                    # node_maping[node] = elem_index[index][0]

        return new_nodes_position, new_elements

    def evaluate_derivatives(self, xi_position: np.ndarray, elements: list=None) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate derivatives at xi_position within each element in *elements* list'

        Parameters:
        -----------

        xi_position (n,3) array of xi positions were to estimate derivatives
        elements: (m) list of elements were to estimate derivatives
        Returns:
        --------

        deriv (n,9): du, dv, duv, duu, dvv, dw, dvw, duw, dww, dudvdw
        """

        if not self.build_mode:
            print(
                "Derivatives are computed in build mode "
                "build model with build_mode=True"
            )
            return

        if elements is None:
            elements = list(range(np.max(self.et_vertex_element_num) + 1))

        num_points = len(elements) * len(xi_position)
        # interpolate field on surface nodes

        der_coeff = np.zeros((num_points, self.NUM_NODES, 10))

        der_e, der_xi = list(), list()

        for et_index in elements:
            for j, xi in enumerate(xi_position):
                xig_index = et_index * len(xi_position) + j

                _, der_coeff[xig_index, :, :], _ = self.evaluate_basis_matrix(
                    xi[0], xi[1], xi[2], et_index, 0, 0, 0
                )

                der_e.append(et_index)
                der_xi.append(xi)

        bxe = np.zeros((num_points, 10))
        bye = np.zeros((num_points, 10))
        bze = np.zeros((num_points, 10))

        for i in range(10):
            bxe[:, i] = np.dot(der_coeff[:, :, i], self.control_mesh[:, 0])
            bye[:, i] = np.dot(der_coeff[:, :, i], self.control_mesh[:, 1])
            bze[:, i] = np.dot(der_coeff[:, :, i], self.control_mesh[:, 2])

        return der_e, der_xi, bxe, bye, bze

    def evaluate_basis_matrix(
        self, s, t, u, elem_number, displacement_s=0, displacement_t=0, displacement_u=0
    ):
        """Evaluates position  and derivatives coefficients (basis function)
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


        """
        if not self.build_mode:
            print(
                "To evaluate gauss points the model should be "
                "read with build_mode=True"
            )
            return

        # Allocate memory
        params_per_element = 32
        pWeights = np.zeros((params_per_element))  # weights
        dWeights = np.zeros((params_per_element, 10))  # weights derivatives

        matrix_coefficient_Bspline_points = np.zeros(params_per_element)
        matrix_coefficient_Bspline_der = np.zeros((params_per_element, 10))
        full_matrix_coefficient_points = np.zeros((self.NUM_NODES))
        full_matrix_coefficient_der = np.zeros((self.NUM_NODES, 10))

        # The b-spline weight of a point is computed giving local coordinates
        # with respect to the 'child patches' elements. The input local
        # coordinates are given with respect to the control grid element

        # Local coordinates of the 'child' patches  are constant and
        # therefore they were precomputed and stored in patch_coordinates
        # and 'fraction` files. They will be later used to interpolate
        # 'patch' local coordinates (within surface element)
        # form 'face' local coordinates (within control grid element)

        # Find projection into endo and epi surfaces

        ps = np.zeros(2)  # local coordinate in the child patch
        pt = np.zeros(2)  # local coordinate in the child patch
        fract = np.zeros(2)
        boundary_value = np.zeros(2)
        b_spline_support = np.ones((2, self.b_spline.shape[1]))
        # select surface vertices associated with the given element number (
        # control mesh)
        # The element number is defined in the coarse mesh
        index_verts = self.et_vertex_element_num[:] == elem_number

        for surface in range(2):  # s= 0 for endo surface and s = 1 for epi
            # surface
            # select vertices from the surface
            index_surface = self.et_vertex_xi[:, 2] == surface
            element_verts_xi = self.et_vertex_xi[
                np.logical_and(index_surface, index_verts), :2
            ]

            if len(element_verts_xi) > 0:
                # find the closest surface point
                elem_tree = cKDTree(element_verts_xi)
                ditance, closest_vertex_id = elem_tree.query([s, t])
                index_surface = np.where(np.logical_and(index_surface, index_verts))[0][
                    closest_vertex_id
                ]

                # translate face to patch coordinates

                if self.fraction[index_surface] != 0:
                    ps[surface] = (
                        s + displacement_s - element_verts_xi[closest_vertex_id, 0]
                    ) / self.fraction[index_surface] + self.patch_coordinates[
                        index_surface, 0
                    ]
                    pt[surface] = (
                        t + displacement_t - element_verts_xi[closest_vertex_id, 1]
                    ) / self.fraction[index_surface] + self.patch_coordinates[
                        index_surface, 1
                    ]
                    b_spline_support[surface] = self.b_spline[index_surface, :]

                    boundary_value[surface] = self.boundary[index_surface]

                    fract[surface] = 1 / self.fraction[index_surface]

            elif elem_number > 166:
                # some surface nodes are not needed for the
                # definition of the biventricular 2D surface therefore they are
                # not include in the surface node matrix. However they are
                # necessary for the 3D interpolation (septum area).
                # these elements are called the phantom points and the
                # corresponding information as the sudivision level ,
                # patch coordinates etc are stored in phantom points array.
                index_phantom = self.phantom_points[:, 0] == elem_number
                elem_phantom_points = self.phantom_points[index_phantom, :]
                elem_vertex_xi = np.stack(
                    (elem_phantom_points[:, 21], elem_phantom_points[:, 22])
                ).T

                elem_tree = cKDTree(elem_vertex_xi)
                ditance, closest_vertex_id = elem_tree.query([s, t])

                if elem_phantom_points[closest_vertex_id, 24] != 0:
                    boundary_value[surface] = elem_phantom_points[closest_vertex_id, 17]
                    fract[surface] = 1 / elem_phantom_points[closest_vertex_id, 24]

                    ps[surface] = (
                        s + displacement_s - elem_phantom_points[closest_vertex_id, 21]
                    ) / elem_phantom_points[
                        closest_vertex_id, 24
                    ] + elem_phantom_points[
                        closest_vertex_id, 18
                    ]
                    pt[surface] = (
                        t + displacement_t - elem_phantom_points[closest_vertex_id, 22]
                    ) / elem_phantom_points[
                        closest_vertex_id, 24
                    ] + elem_phantom_points[
                        closest_vertex_id, 19
                    ]
                    b_spline_support[surface] = elem_phantom_points[
                        closest_vertex_id, 1:17
                    ].astype(int)

        u1 = u + displacement_u
        # normalize s, t coordinates
        control_points = np.concatenate((b_spline_support[0], b_spline_support[1]))
        if len(control_points) < 32:
            print("stop")
        # Uniform B - Splines basis functions
        sWeights = np.zeros((4, 2))
        tWeights = np.zeros((4, 2))
        uWeights = np.zeros(2)
        # Derivatives of the B - Splines basis functions
        ds = np.zeros((4, 2))
        dt = np.zeros((4, 2))
        du = np.zeros(2)
        # Second derivatives of the B - Splines basis functions
        dss = np.zeros((4, 2))
        dtt = np.zeros((4, 2))

        # populate arrays
        for surface in range(2):
            sWeights[:, surface] = basis_function_bspline(ps[surface])
            tWeights[:, surface] = basis_function_bspline(pt[surface])

            ds[:, surface] = der_basis_function_bspline(ps[surface])
            dt[:, surface] = der_basis_function_bspline(pt[surface])

            dss[:, surface] = der2_basis_function_bspline(ps[surface])
            dtt[:, surface] = der2_basis_function_bspline(pt[surface])

            # Adjust the boundaries
            sWeights[:, surface], tWeights[:, surface] = adjust_boundary_weights(
                boundary_value[surface], sWeights[:, surface], tWeights[:, surface]
            )

            ds[:, surface], dt[:, surface] = adjust_boundary_weights(
                boundary_value[surface], ds[:, surface], dt[:, surface]
            )

            dss[:, surface], dtt[:, surface] = adjust_boundary_weights(
                boundary_value[surface], dss[:, surface], dtt[:, surface]
            )

        uWeights[0] = 1 - u1  # linear interpolation
        uWeights[1] = u1  # linear interpolation

        du[0] = -1
        du[1] = 1

        # Weights of the 16 tensors B - spline basis functions and their derivatives
        for k in range(2):
            for i in range(4):
                for j in range(4):
                    index = 16 * k + 4 * i + j
                    pWeights[index] = sWeights[j, k] * tWeights[i, k] * uWeights[k]

                    dWeights[index, 0] = (
                        ds[j, k] * tWeights[i, k] * fract[k] * uWeights[k]
                    )

                    # dScale; % dphi / du = 2 ^ (p * n) * dx, where
                    #  n = level of the patch(0, 1 or 2) and p = order of differentiation.Here
                    #  p = 1 and n = 1 / biv_model.fraction(indx)
                    dWeights[index, 1] = (
                        sWeights[j, k] * dt[i, k] * fract[k] * uWeights[k]
                    )

                    dWeights[index, 2] = (
                        ds[j, k] * dt[i, k] * (fract[k] ** 2) * uWeights[k]
                    )

                    dWeights[index, 3] = (
                        dss[j, k] * tWeights[i, k] * (fract[k] ** 2) * uWeights[k]
                    )

                    dWeights[index, 4] = (
                        sWeights[j, k] * dtt[i, k] * (fract[k] ** 2) * uWeights[k]
                    )

                    dWeights[index, 5] = sWeights[j, k] * tWeights[i, k] * du[k]

                    dWeights[index, 6] = sWeights[j, k] * dt[i, k] * du[k] * fract[k]

                    dWeights[index, 7] = ds[j, k] * tWeights[i, k] * du[k] * fract[k]

                    dWeights[index, 8] = 0  # % linear interpolation --> duu = 0
                    dWeights[index, 9] = ds[j, k] * dt[i, k] * (fract[k] ** 2) * du[k]

            # add weights
        for i in range(32):
            matrix_coefficient_Bspline_points[i] = pWeights[i]
            full_matrix_coefficient_points = (
                full_matrix_coefficient_points
                + pWeights[i] * self.local_matrix[int(control_points[i]), :]
            )
            for k in range(10):
                matrix_coefficient_Bspline_der[i, k] = dWeights[i, k]
                full_matrix_coefficient_der[:, k] = (
                    full_matrix_coefficient_der[:, k]
                    + dWeights[i, k] * self.local_matrix[int(control_points[i]), :]
                )

        return (
            full_matrix_coefficient_points,
            full_matrix_coefficient_der,
            control_points,
        )

    def compute_local_cs(self, position: list, element_number: int=None) -> None:
        """Computes local coordinates system at any point of the subdivision
        surface. x1 - circumferential direction, x2- longitudinal direction,
        x3- transmural direction

        Parameters
        ------------

        `position` list of (3,1) arrays[float] with xi coordinates

        `element_number` index of the control elements (coarse mesh). if non
        is specified the local cs is computed for all elements

        Return
        -------

        """
        # todo method not tested
        if not self.build_mode:
            print(
                "field evaluation is performed in build mode "
                "build model with build_mode=True"
            )
            return

        if element_number is None:
            element_number = np.array(range(np.max(self.et_vertex_element_num)))

        dxi = 0.01

        basis_matrix = np.zeros((len(element_number) * len(position), self.NUM_NODES))
        basis_matrix_dx1 = np.zeros(
            (len(element_number) * len(position), self.NUM_NODES)
        )
        basis_matrix_dx2 = np.zeros(
            (len(element_number) * len(position), self.NUM_NODES)
        )
        basis_matrix_dx3 = np.zeros(
            (len(element_number) * len(position), self.NUM_NODES)
        )
        for et_indx, control_et in enumerate(element_number):
            for j, xi in enumerate(position):
                g_indx = et_indx * len(position) + j
                # basis matrix for node position
                basis_matrix[g_indx, :], _, _ = self.evaluate_basis_matrix(
                    xi[0], xi[1], xi[2], control_et, 0, 0, 0
                )

                # basis matrix for node position with increment of dxi in x1
                # direction
                basis_matrix_dx1[g_indx, :], _, _ = self.evaluate_basis_matrix(
                    xi[0], xi[1], xi[2], control_et, dxi, 0, 0
                )

                # basis matrix for node position with increment of dxi in x2
                # direction
                basis_matrix_dx2[g_indx, :], _, _ = self.evaluate_basis_matrix(
                    xi[0], xi[1], xi[2], control_et, 0, dxi, 0
                )

                # basis matrix for node position with increment of dxi in x1
                # direction
                basis_matrix_dx3[g_indx, :], _, _ = self.evaluate_basis_matrix(
                    xi[0], xi[1], xi[2], control_et, 0, 0, dxi
                )

                # todo The local cs is computed by 1) dot product with control
                # matrix to compute node position. 2) subtract points to
                # compute the corresponding vectors

    def evaluate_surface_field(self, field: np.ndarray, vertex_map: list) -> np.ndarray:
        """Evaluate field at the each surface points giving a sparse field
        defined at a subset of surface points.

        Notes
        ------

        Input surface field need to have more than 388 points evenly spread
        on the subdivided surface. A good choice is to define the filed at
        each surface point with each xi coordinate equal to 0 or 1

        Parameters
        -----------

        `field` (NUM_NODES, k) array[float] field to be interpolated

        `vertex_map` (NUM_NODES,1) array[ints] nodes id where the field is
        defined

        Returns
        --------

        `interpolated_field` (NUM_SURFACE_NODES, k) array[float] interpolated
        field at each surface node

        """
        if not self.build_mode:
            print(
                "field evaluation is performed in build mode "
                "build model with build_mode=True"
            )
            return

        basis_matrix = np.zeros((len(vertex_map), self.NUM_NODES))

        # first estimate the control points from the known fields
        for v_index, v in enumerate(vertex_map):
            xi = self.et_vertex_xi[v]
            et_index = self.et_vertex_element_num[v]

            basis_matrix[v_index, :], _, _ = self.evaluate_basis_matrix(
                xi[0], xi[1], xi[2], et_index, 0, 0, 0
            )

        control_points = np.linalg.solve(basis_matrix, field)

        # interpolate field on surface nodes
        basis_matrix = np.zeros((self.NUM_SURFACE_NODES, self.NUM_NODES))
        for et_index in range(np.max(self.et_vertex_element_num) + 1):
            xig_index = np.where(self.et_vertex_element_num == et_index)[0]
            xig = self.et_vertex_xi[xig_index]
            for j, xi in enumerate(xig):
                basis_matrix[xig_index[j], :], _, _ = self.evaluate_basis_matrix(
                    xi[0], xi[1], xi[2], et_index, 0, 0, 0
                )

        interpolated_field = np.dot(basis_matrix, control_points)

        return interpolated_field

    def evaluate_field(self, field: np.ndarray, vertex_map: list, position: np.ndarray, elements: list=None) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates field at the each xi position within a element of the
        control grid

        Notes
        ------

        Input surface field need to have more than 388 points evenly spread
        on the subdivided surface. A good choice is to define the filed at
        each surface point with each xi coordinate equal to 0 or 1

        Parameters
        -----------

        `field` (`NUM_NODES`,k) matrix[float] field to be interpolated

        `vertex_map` ('NUM_NODES`) vector[ints] nodes id where the field is
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

        """
        if not self.build_mode:
            print(
                "field evaluation is performed in build mode "
                "build model with build_mode=True"
            )
            return
        if elements is None:
            elements = list(range(np.max(self.et_vertex_element_num) + 1))

        basis_matrix = np.zeros((len(vertex_map), self.NUM_NODES))

        # first estimate the control points from the known fiels
        for v_index, v in enumerate(vertex_map):
            xi = self.et_vertex_xi[v]
            et_index = self.et_vertex_element_num[v]

            basis_matrix[v_index, :], _, _ = self.evaluate_basis_matrix(
                xi[0], xi[1], xi[2], et_index, 0, 0, 0
            )

        control_points = np.linalg.solve(basis_matrix, field)

        # interpolate field on surface nodes

        basis_matrix = np.zeros((len(elements) * len(position), self.NUM_NODES))
        for i, et_index in enumerate(elements):
            for j, xi in enumerate(position):
                index = i * len(position) + j
                basis_matrix[index, :], _, _ = self.evaluate_basis_matrix(
                    xi[0], xi[1], xi[2], et_index, 0, 0, 0
                )

        interpolated_field = np.dot(basis_matrix, control_points)
        points = np.dot(basis_matrix, self.control_mesh)

        return points, interpolated_field