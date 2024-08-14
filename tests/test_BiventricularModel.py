from bivme.fitting import BiventricularModel as bm
from bivme import MODEL_RESOURCE_DIR
import pytest

@pytest.mark.parametrize("test_input,expected",[
            (True, True),
            (False, False)
        ])

##TODO test if build is True, check size for other matrices
def test_init_build_mode(test_input, expected):
    model = bm.BiventricularModel(MODEL_RESOURCE_DIR, build_mode=test_input)
    assert model.build_mode is expected


def test_default_label_and_build():

    model = bm.BiventricularModel(MODEL_RESOURCE_DIR)
    assert model.build_mode == False
    assert model.label == "default"
    assert model.NUM_NODES == 388
    assert model.NUM_ELEMENTS == 187
    assert model.NUM_SURFACE_NODES == 5810
    assert model.APEX_INDEX == 50
    assert model.NUM_GAUSSIAN_POINTS == 5049
    assert model.NUM_SUBDIVIDED_FACES == 11760
    assert model.NUM_NODES_THRU_WALL == 160
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

    # if build mode is on
    #assert model.et_vertex_xi.shape == (model.NUM_SURFACE_NODES, 3)
    #assert model.b_spline.shape ==
    #assert model.control_et_indices.shape ==
    #assert model.phantom_points.shape ==
    #assert model.patch_coordinates.shape ==
    #assert model.fraction.shape ==
    #assert model.local_matrix.shap ==
