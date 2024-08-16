#from bivme.fitting import BiventricularModel as bm

from bivme.fitting.BiventricularModel import BiventricularModel
from bivme.fitting.GPDataSet import GPDataSet
from bivme import MODEL_RESOURCE_DIR, TEST_RESOURCE_DIR
from bivme.fitting.surface_enum import Surface
import pytest
import numpy as np
from bivme.fitting.surface_enum import Surface

test_model = BiventricularModel(MODEL_RESOURCE_DIR, build_mode=True)

@pytest.mark.parametrize("test_build_input,expected_build",[
            (True, True),
            (False, False)
        ])

def test_init_build_mode(test_build_input, expected_build):
    model = BiventricularModel(MODEL_RESOURCE_DIR, build_mode=test_build_input)
    assert model.build_mode is expected_build

    assert model.label == "default"
    assert model.NUM_NODES == 388
    assert model.NUM_ELEMENTS == 187
    assert model.NUM_SURFACE_NODES == 5810
    assert model.APEX_INDEX == 3261#50
    assert model.NUM_GAUSSIAN_POINTS == 5049
    assert model.NUM_SUBDIVIDED_FACES == 11760
    assert model.NUM_NODES_THRU_WALL == 160
    assert model.NUM_LOCAL_POINTS == 12509

    assert model.control_mesh.shape == (model.NUM_NODES, 3)
    assert model.matrix.shape == (model.NUM_SURFACE_NODES, model.NUM_NODES)
    assert model.et_pos.shape == (model.NUM_SURFACE_NODES, 3)
    assert model.jac_11.shape == (11968, model.NUM_NODES)
    assert model.jac_12.shape == (11968, model.NUM_NODES)
    assert model.jac_13.shape == (11968, model.NUM_NODES)
    assert model.et_indices.shape == (model.NUM_SUBDIVIDED_FACES, 3)
    assert model.et_vertex_element_num.shape == (model.NUM_SURFACE_NODES,)
    assert model.basis_matrix.shape == (model.NUM_SURFACE_NODES, model.NUM_NODES)
    assert model.gtstsg_x.shape == (model.NUM_NODES, model.NUM_NODES)
    assert model.gtstsg_y.shape == (model.NUM_NODES, model.NUM_NODES)
    assert model.gtstsg_z.shape == (model.NUM_NODES, model.NUM_NODES)
    assert model.et_indices_thru_wall.shape == (model.NUM_NODES_THRU_WALL, 3)
    assert model.et_indices_epi_lvrv.shape == (model.surface_start_end[3][1] - model.surface_start_end[3][0] + 1, 3)
    assert model.mbder_dx.shape == (model.NUM_GAUSSIAN_POINTS, model.NUM_NODES)
    assert model.mbder_dy.shape == (model.NUM_GAUSSIAN_POINTS, model.NUM_NODES)
    assert model.mbder_dz.shape == (model.NUM_GAUSSIAN_POINTS, model.NUM_NODES)

    if expected_build:
        assert model.fraction.shape == (model.NUM_SURFACE_NODES, 1)
        assert model.control_et_indices.shape == (model.NUM_ELEMENTS, 8)
        assert model.et_vertex_xi.shape == (model.NUM_SURFACE_NODES, 3)
        assert model.b_spline.shape == (model.NUM_SURFACE_NODES, 16)
        assert model.patch_coordinates.shape == (model.NUM_SURFACE_NODES, 3)
        assert model.phantom_points.shape == (500, 25)

def test_get_node():
    nodes = test_model.get_nodes()
    assert np.array_equal(nodes, test_model.et_pos)

def test_get_control_mesh_nodes():
    control_nodes = test_model.get_control_mesh_nodes()
    assert np.array_equal(control_nodes, test_model.control_mesh)

@pytest.mark.parametrize("test_surface, expected_surface_indices",[
            (Surface.LV_ENDOCARDIAL, test_model.surface_start_end[0, :]),
            (Surface.RV_SEPTUM, test_model.surface_start_end[1, :]),
            (Surface.RV_FREEWALL, test_model.surface_start_end[2, :]),
            (Surface.EPICARDIAL, test_model.surface_start_end[3, :]),
            (Surface.MITRAL_VALVE, test_model.surface_start_end[4, :]),
            (Surface.AORTA_VALVE, test_model.surface_start_end[5, :]),
            (Surface.TRICUSPID_VALVE, test_model.surface_start_end[6, :]),
            (Surface.PULMONARY_VALVE, test_model.surface_start_end[7, :])
        ])
def test_get_surface_start_end_index(test_surface, expected_surface_indices):
    assert np.array_equal(test_model.get_surface_start_end_index(test_surface), expected_surface_indices)

@pytest.mark.parametrize("test_surface, expected_vertex_indices",[
            (Surface.LV_ENDOCARDIAL, test_model.et_vertex_start_end[0, :]),
            (Surface.RV_SEPTUM, test_model.et_vertex_start_end[1, :]),
            (Surface.RV_FREEWALL, test_model.et_vertex_start_end[2, :]),
            (Surface.EPICARDIAL, test_model.et_vertex_start_end[3, :]),
            (Surface.MITRAL_VALVE, test_model.et_vertex_start_end[4, :]),
            (Surface.AORTA_VALVE, test_model.et_vertex_start_end[5, :]),
            (Surface.TRICUSPID_VALVE, test_model.et_vertex_start_end[6, :]),
            (Surface.PULMONARY_VALVE, test_model.et_vertex_start_end[7, :]),
            (Surface.RV_INSERT, test_model.et_vertex_start_end[8, :]),
            (Surface.APEX, [test_model.APEX_INDEX, test_model.APEX_INDEX])
        ])
def test_get_surface_vertex_start_end_index(test_surface, expected_vertex_indices):
    assert np.array_equal(test_model.get_surface_vertex_start_end_index(test_surface), expected_vertex_indices)

@pytest.mark.parametrize("model_path, expected_output",[
            (MODEL_RESOURCE_DIR / "model.txt", True),
            (TEST_RESOURCE_DIR / "diffeomorphism_data" / "non_diffeomorphic_case_0.txt", False),
            (TEST_RESOURCE_DIR / "diffeomorphism_data" / "non_diffeomorphic_case_1.txt", False),
            (TEST_RESOURCE_DIR / "diffeomorphism_data" / "diffeomorphism_case_0.txt", True),
            (TEST_RESOURCE_DIR / "diffeomorphism_data" / "non_diffeomorphic_case_2.txt", False)
        ])
def test_is_diffeomorphic(model_path, expected_output):
    updated_position = np.loadtxt(model_path, delimiter='\t').astype(float)
    assert test_model.is_diffeomorphic(updated_position, 0.0) == expected_output

def test_update_control_mesh():
    updated_control_mesh = test_model.control_mesh
    updated_control_mesh[[1, 50, 5, 8],:] += updated_control_mesh[[1,50,5,8],:]
    test_model.update_control_mesh(updated_control_mesh)
    assert np.array_equal(test_model.control_mesh, updated_control_mesh)
    updated_control_mesh[[1, 50, 5, 8],:] /= 2
    test_model.update_control_mesh(updated_control_mesh)
    assert np.array_equal(test_model.control_mesh, updated_control_mesh)

def test_get_scaling():

    scaling_factors = [0.5, 3, 10, 0.00001]
    gp_dataset = GPDataSet()

    for factor in scaling_factors:
        # create fake GPDataSet - only valves and apex are needed for scaling
        gp_dataset.apex = factor * test_model.et_pos[test_model.APEX_INDEX,]
        gp_dataset.mitral_centroid = factor * test_model.et_pos[
            test_model.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1],]
        gp_dataset.tricuspid_centroid = factor * test_model.et_pos[
            test_model.get_surface_vertex_start_end_index(Surface.TRICUSPID_VALVE)[1],]

        scale = test_model.get_scaling(gp_dataset)
        assert scale == factor

def test_get_translation():

    translation_vector = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0],[0, 0, 10], [10, 10, 10]])
    gp_dataset = GPDataSet()

    for translation in translation_vector:
        # create fake GPDataSet - only valves and apex are needed for scaling
        gp_dataset.apex = test_model.et_pos[test_model.APEX_INDEX,] + translation
        gp_dataset.mitral_centroid = test_model.et_pos[
            test_model.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1],] + translation
        gp_dataset.tricuspid_centroid = test_model.et_pos[
            test_model.get_surface_vertex_start_end_index(Surface.TRICUSPID_VALVE)[1],] + translation

        model_translation = test_model.get_translation(gp_dataset)
        assert np.array_equal(model_translation,translation)

    #test_model = bm.BiventricularModel(MODEL_RESOURCE_DIR)

    #transformation_matrix = np.array([[],[],[],[]])

#def test_get_translation():

#def test_get_rotation():

#    transformation_matrix_x = np.array([[],[],[],[]])
#    transformation_matrix_y = np.array([[],[],[],[]])
#    transformation_matrix_z = np.array([[],[],[],[]])
#    transformation_matrix_xyz = np.array([[],[],[],[]])

# def test_update_pose_and_scale():
# def evaluate_field(self, field, vertex_map, position, elements=None)
# def evaluate_surface_field
# def compute_local_cs
# def evaluate_basis_matrix
# def evaluate_derivatives
# def extract_linear_hex_mesh
# def compute_data_xi
# def get_surface_faces
# get_intersection_with_dicom_image
# get_intersection_with_plane
# def plot_surface