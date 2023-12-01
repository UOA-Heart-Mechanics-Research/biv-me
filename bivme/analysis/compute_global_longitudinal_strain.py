# Author: Anna Mira
# Reviewed by Laura Dal Toso on 7/08/2023

# This script computes the GLS from the models output by the BIVFitting code


import os
import numpy as np
import pandas as pd
import sys 
sys.path.append('../Preprocessing/mesh_tools/')
from geometric_tools import *
from mesh import *
import visualization as viewer
#import mesh_tools.visualization as viewer
import time

from mayavi import mlab
import csv
import pylab as pl


def GL_strain( folder,output_file):

    '''
    # Author: ldt
    # Date: 18/08/22

    This functions measures various strain metrics, from the models fitted at ES and ED.
    Input: 
        - folder: folder where the Model.txt files are saved 
        - output_file: csv file where the srtain measures should be saved  

    '''

    case =  os.path.basename(os.path.normpath(folder))
    print('case: ', case )

    input_path_model = '../../BiV_Modelling/model/'
    subdivision_matrix_file = os.path.join(input_path_model,
                                                "subdivision_matrix.txt")
    model_file = os.path.join(input_path_model, 'model.txt')
    elements_file = os.path.join(input_path_model,'ETIndicesSorted.txt') #3 vertices for each face
    material_file = os.path.join(input_path_model,'ETIndicesMaterials.txt') #face indices

    ls_points_file = './InputFiles/ls_points.txt'
    plot_figure = True
    

    if not os.path.exists(subdivision_matrix_file):
        ValueError('Missing subdivision_matrix.txt')
    if not os.path.exists(model_file):
        ValueError('Missing model.txt')
    if not os.path.exists(elements_file):
        ValueError('Missing ETIndicesSorted.txt')
    if not os.path.exists(material_file):
        ValueError('Missing ETIndicesMaterials.txt')


    ls_points = pd.read_table(ls_points_file, sep = ' ')
    ################## read model ########################
    subdivision_matrix = (np.loadtxt(subdivision_matrix_file)).astype(float)
    control_points = np.loadtxt(model_file).astype(float)
    faces = np.loadtxt(elements_file).astype(int)-1
    vertices =  np.dot(subdivision_matrix, control_points)
    mat = np.loadtxt(material_file, dtype='str')



    ## convert labels to integer correspondig to the sorted list
    # of unique labels types
    unique_material = np.unique(mat[:,1])
    materials  = np.zeros(mat.shape)

    for index,m in enumerate(unique_material):
        face_index = mat[:,1] == m
        materials[face_index,0] = mat[face_index,0].astype(int)
        materials[face_index, 1] = [index] *np.sum(face_index)


    model = Mesh('mesh')
    model.set_nodes(vertices*10)
    model.set_elements(faces)
    model.set_materials(materials[:,0], materials[:,1])

    # Select RV endocardial
    RV_ENDO = model.get_mesh_component([6,9,10,11], reindex_nodes=False)
    RV_ENDO.materials = np.zeros_like(RV_ENDO.materials)
    RV_SEPTUM = model.get_mesh_component([10], reindex_nodes=False)
    RV_SEPTUM.materials = np.zeros_like(RV_SEPTUM.materials)
    LV_ENDO = model.get_mesh_component([0,2,4], reindex_nodes=False)
    LV_ENDO.materials = np.zeros_like(LV_ENDO.materials)

    if plot_figure:

        fig = viewer.Figure('model', size=(600, 600))
        fig.plot_mesh('wire', model, opacity=1, line_colour=(0, 0.2, 0.3),
                    mode = 'wireframe')
        fig.plot_mesh('LV_ENDO', LV_ENDO, opacity=0.99, line_opacity= 0.3,
                    face_colour=(0.8, 0.9, 0.8))
        fig.plot_mesh('RV_ENDO', RV_ENDO, opacity=0.99, line_opacity= 0.3,
                    face_colour=(0.8, 1, 0.9))
        fig.plot_mesh('RV_SEPTUM', RV_SEPTUM, opacity=0.99, line_opacity= 0.3,
                    face_colour=(0.8, 1,   0.9))
        


    # Computation of strain 
    lv_lax_verts = []
    rv_lax_verts = []
    rvs_lax_verts = []

    region = ['4CH', '3CH']

    for slice in range(len(region)):

        # extract LV LAX slice
        index = np.logical_and(ls_points.Surface == 'LV', ls_points.Level ==
                            region[slice])
        verts = ls_points.ENDO[index]
        lv_lax_verts.append(verts)

        if plot_figure:
            fig.plot_points('lvslice{0}'.format(slice),
                            model.nodes[verts], size=2,
                        color=(1, 0, 0))

        # extract LAX RV slice
        index = np.logical_and(ls_points.Surface == 'RV', ls_points.Level ==
                            region[slice])
        verts = ls_points.ENDO[index]

        rv_lax_verts.append(verts)
        if plot_figure:
            fig.plot_points('rvslice{0}'.format(slice),
                            model.nodes[verts], size=1.5,
                        color=(0.6, 0, 0.6))

        # extract RVS LAX slice
        index = np.logical_and(ls_points.Surface == 'RVS', ls_points.Level ==
                            region[slice])
        verts = ls_points.ENDO[index]
        rvs_lax_verts.append(np.array(verts))

        if plot_figure:
            fig.plot_points('rvsslice{0}'.format(slice),
                            model.nodes[verts], size=1.5,
                        color=(0, 0, 1))

            mlab.show()
            #mlab.savefig('./results/mlab_contour3d.png')



    models  = [k for k in np.sort(list(os.listdir(folder))) if 'Model' in k]
    case = os.path.basename(os.path.normpath(folder))

    if len(models)>0:
        
        try:
            
            for frame in models: 
                if 'ES' in frame:
                    node_file_ES = os.path.join(folder, frame)
                else:
                    node_file_ED = os.path.join(folder, frame)

            if not (os.path.exists(node_file_ES)):
                raise ValueError
            if not (os.path.exists(node_file_ED)):
                raise ValueError

            fitted_nodes_ES = np.dot(subdivision_matrix,
                                    np.loadtxt(node_file_ES, delimiter=',',skiprows=1).astype(float))
            fitted_nodes_ED = np.dot(subdivision_matrix,
                                np.loadtxt(node_file_ED,delimiter=',',skiprows=1).astype(float)  )

            lv_ls = []
            rv_ls = []
            rvs_ls =[]
            
            
            for i in range (0,len(region)):
                
                #         compute lv local strain
                l_ed = np.linalg.norm(fitted_nodes_ED[lv_lax_verts[i][1:]] - \
                                    fitted_nodes_ED[lv_lax_verts[i][:-1]], axis=1)
                l_es = np.linalg.norm(fitted_nodes_ES[lv_lax_verts[i][1:]] - \
                                    fitted_nodes_ES[lv_lax_verts[i][:-1]], axis = 1)

                if plot_figure:
                    fig.plot_points('rvslice{0}'.format(slice),l_ed, size=1.5,
                                color=(0.6, 0, 0.6))    

                mlab.show()   

                lv_ls.append( (np.sum(l_es)-np.sum(l_ed))/np.sum(l_ed))

            #        compute rv local strain
                l_ed = np.linalg.norm(fitted_nodes_ED[rv_lax_verts[i][1:]] - \
                                    fitted_nodes_ED[rv_lax_verts[i][:-1]], axis=1)
                l_es = np.linalg.norm(fitted_nodes_ES[rv_lax_verts[i][1:]] - \
                                    fitted_nodes_ES[rv_lax_verts[i][:-1]], axis = 1)
                rv_ls.append( (np.sum(l_es)-np.sum(l_ed))/np.sum(l_ed))

                #         compute rv septum local strain
                l_ed = np.linalg.norm(fitted_nodes_ED[rvs_lax_verts[i][1:]] - \
                                    fitted_nodes_ED[rvs_lax_verts[i][:-1]],
                                    axis=1)
                l_es = np.linalg.norm(fitted_nodes_ES[rvs_lax_verts[i][1:]] - \
                                    fitted_nodes_ES[rvs_lax_verts[i][:-1]],
                                    axis=1)
                rvs_ls.append((np.sum(l_es) - np.sum(l_ed)) / np.sum(l_ed))
            
            # compute general longitudinal strain for LV

            lv_gls_4CH = lv_ls[0]
            lv_gls_3CH = lv_ls[1]
            lv_gls = np.mean(lv_ls)
            rv_gls_4CH = rv_ls[0]
            rv_gls_3CH =rv_ls[1]
            rv_gls = np.mean(rv_ls)
            rvs_gls_4CH = rvs_ls[0]
            rvs_gls_3CH = rvs_ls[1]
            rvs_gls = np.mean(rvs_ls)

            with open(output_file, 'a',newline="") as f:
                writer = csv.writer(f)
                writer.writerow([case, lv_gls_4CH, lv_gls_3CH, lv_gls, rv_gls_4CH, rv_gls_3CH,
                    rv_gls, rvs_gls_4CH, rvs_gls_3CH, rvs_gls])
                    
            return {'case':case, 'lv_gls_4CH':lv_gls_4CH, 'lv_gls_3CH':lv_gls_3CH, 
                'lv_gls':lv_gls,'rv_gls_4CH':rv_gls_4CH,'rv_gls_3CH':rv_gls_3CH, 'rv_gls':rv_gls,
                'rvs_gls_4CH':rvs_gls_4CH, 'rvs_gls_3CH':rvs_gls_3CH, 'rvs_gls':rvs_gls}
        
        
        except:
            pass 


if __name__ == '__main__':

   
    path_to_models = './results'
    output_file = './results/global_longitudinal_strain.csv'
    

    # listdir in path_model   
    EDES_folders = sorted(list(os.listdir(path_to_models)))
    cases = [patient for patient in EDES_folders if os.path.isdir(
        os.path.join(path_to_models,patient)) == True]
    
    fieldnames = ['name','lv_gls_4CH','lv_gls_3CH','lv_gls','rv_gls_4CH', 'rv_gls_3CH',
        'rv_gls' ,'rvs_gls_4CH','rvs_gls_3CH','rvs_gls']

    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames= fieldnames)
        writer.writeheader() 


    print('files in EDES folder', cases)

    # use the following if you need to visualise one patient. 
    # Change the number in squared brackets depending on which patient you want to visualise.
    # also set plot_figure = True
    #result_1case = find_strain(os.path.join(path_to_models, folders[1]), output_file)

    # use the following line if you want the final excel file for all patients
    all_results = GL_strain(os.path.join(path_to_models,cases[0]), output_file)
    #if os.path.isdir(os.path.join(path_to_models,patient)) == True]

