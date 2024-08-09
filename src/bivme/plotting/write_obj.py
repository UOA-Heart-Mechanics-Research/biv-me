import os
import pyvista as pv
from pathlib import Path

from echonet.dataproc.prepare_data import load_vtk_mesh
from bivme.fitting.perform_fit import write_vtk_surface


if __name__ == "__main__":
    
    vtkdir = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Debbie\collaborations\chicago-rv-mesh\analysis\pilot-obj"
    objdir = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Debbie\collaborations\chicago-rv-mesh\analysis\pilot-obj"
    
    # get all files in vtkdir
    vtkfiles = next(os.walk(vtkdir))[2]
    
    for vf in vtkfiles:
        
        mesh = pv.read(Path(vtkdir, vf))
        pl = pv.Plotter()
        _ = pl.add_mesh(mesh)
        pl.export_obj(Path(objdir, f'{Path(vf).stem}.obj'))

        print(f'Converted {vf} to .obj')
        