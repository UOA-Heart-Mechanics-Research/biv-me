from __future__ import annotations
import numpy as np
import pyvista as pv


from bivme import MODEL_RESOURCE_DIR
from bivme.meshing.mesh import Mesh
from bivme.fitting.BiventricularModel import BiventricularModel

biv_model_folder = MODEL_RESOURCE_DIR

#import pymeshfix


biventricular_model = BiventricularModel(biv_model_folder, collision_detection = True)



biventricular_model.detect_collision()

control_points = np.loadtxt('./collision_mesh.txt', delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)
biventricular_model.update_control_mesh(control_points)

biventricular_model.detect_collision()

#self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)



#lv_endo = model.get_mesh_component([0, 2, 4], reindex_nodes=False)
#lv_faces = lv_endo.elements
#lv_faces = np.pad(lv_faces, ((0, 0), (1, 0)), 'constant', constant_values=3)
#lv_mesh = pv.PolyData(lv_endo.nodes, lv_faces)
#
#pl = pv.Plotter()
#pl.enable_hidden_line_removal()
#pl.add_mesh(sphere0, scalars='collisions', show_scalar_bar=False, cmap='bwr')
#pl.camera_position = 'xz'
#pl.add_mesh(sphere1, style='wireframe', color='green', line_width=5)
#
## for this example
#pl.open_gif("collision_movie.gif")
#
## alternatively, to disable movie generation:
## pl.show(auto_close=False, interactive=False)
#
#delta_x = 0.05
#for _ in range(int(2 / delta_x)):
#    sphere1.translate([delta_x, 0, 0], inplace=True)
#    col, n_contacts = sphere0.collision(sphere1)
#
#    collision_mask = np.zeros(sphere0.n_cells, dtype=bool)
#    if n_contacts:
#        collision_mask[col['ContactCells']] = True
#    sphere0['collisions'] = collision_mask
#    pl.write_frame()
#
#    # alternatively, disable movie plotting and simply render the image
#    # pl.render()
#
#pl.close()
