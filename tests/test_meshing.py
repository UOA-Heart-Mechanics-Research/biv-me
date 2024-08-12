from bivme.meshing import mesh
import numpy as np

def test_get_volume():

    model = mesh.Mesh('mesh')

    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1]])

    faces = np.array([[0, 2, 1],
                      [0, 3, 2],
                      [6, 4, 5],
                      [7, 4, 6],
                      [5, 1, 2],
                      [6, 5, 2],
                      [4, 3, 0],
                      [4, 7, 3],
                      [6, 2, 7],
                      [7, 2, 3],
                      [4, 1, 5],
                      [4, 0, 1]])

    model.set_nodes(vertices)
    model.set_elements(faces)
    volume = model.get_volume()

    assert volume == 0.001