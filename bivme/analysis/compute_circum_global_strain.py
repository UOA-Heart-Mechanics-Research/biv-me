# Author: Anna Mira
# Reviewed by Laura Dal Toso on 18/08/2022

# This script computes the global circumferential strain from the models output by the BIVFitting code


import os
import numpy as np
import pandas as pd
import time
import csv
import sys 
sys.path.append('./mesh_tools/')

from mesh_tools import Mesh
from mesh_tools import *

import visualization as viewer
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def GC_strain( folder,output_file):
    
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

    
    input_path_model = '../BiV_Modelling_v2/model/'
    subdivision_matrix_file = os.path.join(input_path_model,
                                                "subdivision_matrix.txt")
    model_file = os.path.join(input_path_model, 'model.txt')
    elements_file = os.path.join(input_path_model,'ETIndicesSorted.txt')
    material_file = os.path.join(input_path_model,'ETIndicesMaterials.txt')
    circum_points_file = './InputFiles/cs_points.txt'

    plot_figure = False

    if not os.path.exists(subdivision_matrix_file):
        ValueError('Missing subdivision_matrix.txt')
    if not os.path.exists(model_file):
        ValueError('Missing model.txt')
    if not os.path.exists(elements_file):
        ValueError('Missing ETIndicesSorted.txt')
    if not os.path.exists(material_file):
        ValueError('Missing ETIndicesMaterials.txt')


    cs_points = pd.read_table(circum_points_file, sep = ' ')
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

    #print('materials', unique_material)

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
    lv_sax_verts = []
    rv_sax_verts = []
    rvs_sax_verts = []
    region = ['apex','mid','basal']

    for slice_index in range(3):

        index = np.logical_and(cs_points.Surface == 'LV', cs_points.Level ==
                            region[slice_index])
        verts = cs_points.ENDO[index]
        points = model.nodes[verts]
        lv_sax_verts.append(verts)
        if plot_figure:
            fig.plot_points('lvslice{0}'.format(slice_index),
                            model.nodes[verts], size=2,
                        color=(1, 0, 0))

        # extract RV SAX slice
        index = np.logical_and(cs_points.Surface == 'RV', cs_points.Level ==
                            region[slice_index])
        verts = cs_points.ENDO[index]
        rv_sax_verts.append(verts)
        if plot_figure:
            fig.plot_points('rvslice{0}'.format(slice_index),
                            model.nodes[verts], size=1.5,
                            color=(0.6, 0, 0.6))

        # extract RVS SAX slice
        index = np.logical_and(cs_points.Surface == 'RVS', cs_points.Level ==
                            region[slice_index])
        verts = cs_points.ENDO[index]
        rvs_sax_verts.append(np.array(verts))

        if plot_figure:
            fig.plot_points('rvsslice{0}'.format(slice_index),
                            model.nodes[verts], size=1.5,
                        color=(0, 0, 1))
            mlab.show()
    
    models  = [k for k in np.sort(list(os.listdir(folder))) if 'Model' in k]
    case = os.path.basename(os.path.normpath(folder))


    if len(models)>0:
        try:
            for frame in models: 
                
                if 10<int(frame[-7:-4])<30:
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
                                np.loadtxt(node_file_ED, delimiter=',',skiprows=1).astype(float)  )
            lv_cs = []
            rv_cs = []
            rvs_cs =[]
            for i in range(3):
                #         compute lv local circumferential stress
                l_ed = np.linalg.norm(fitted_nodes_ED[lv_sax_verts[i][1:]] - \
                    fitted_nodes_ED[lv_sax_verts[i][:-1]], axis=1)
                l_es = np.linalg.norm(fitted_nodes_ES[lv_sax_verts[i][1:]] - \
                    fitted_nodes_ES[lv_sax_verts[i][:-1]], axis = 1)
                lv_cs.append( (np.sum(l_es)-np.sum(l_ed))/np.sum(l_ed))

                #        compute rv local circumferential stress
                l_ed = np.linalg.norm(fitted_nodes_ED[rv_sax_verts[i][1:]] - \
                    fitted_nodes_ED[rv_sax_verts[i][:-1]], axis=1)
                l_es = np.linalg.norm(fitted_nodes_ES[rv_sax_verts[i][1:]] - \
                    fitted_nodes_ES[rv_sax_verts[i][:-1]], axis = 1)
                rv_cs.append( (np.sum(l_es)-np.sum(l_ed))/np.sum(l_ed))

                #         compute rv septum local circumferential stress
                l_ed = np.linalg.norm(fitted_nodes_ED[rvs_sax_verts[i][1:]] - \
                                    fitted_nodes_ED[rvs_sax_verts[i][:-1]],
                                    axis=1)
                l_es = np.linalg.norm(fitted_nodes_ES[rvs_sax_verts[i][1:]] - \
                                    fitted_nodes_ES[rvs_sax_verts[i][:-1]],
                                    axis=1)
                rvs_cs.append((np.sum(l_es) - np.sum(l_ed)) / np.sum(l_ed))


            lv_gcs_apex = lv_cs[0]
            lv_gcs_mid = lv_cs[1]
            lv_gcs_basal = lv_cs[2]
            lv_gcs = np.mean(lv_cs)

            rv_gcs_apex = rv_cs[0]
            rv_gcs_mid = rv_cs[1]
            rv_gcs_basal = rv_cs[2]
            rv_gcs = np.mean(rv_cs)

            rvs_gcs_apex = rvs_cs[0]
            rvs_gcs_mid = rvs_cs[1]
            rvs_gcs_basal = rvs_cs[2]
            rvs_gcs = np.mean(rvs_cs)


            with open(output_file, 'a',newline="") as f:
                writer = csv.writer(f)
                writer.writerow([case, lv_gcs_apex, lv_gcs_mid, lv_gcs_basal, lv_gcs, rv_gcs_apex, 
                    rv_gcs_mid, rv_gcs_basal, rv_gcs, rvs_gcs_apex,rvs_gcs_mid,rvs_gcs_basal,rvs_gcs])
            
            return {'case':case, 'lv_gcs_apex':lv_gcs_apex, 'lv_gcs_mid':lv_gcs_mid, 'lv_gcs_basal':lv_gcs_basal, 'lv_gcs':lv_gcs, 
                    'rv_gcs_apex':rv_gcs_apex, 'rv_gcs_mid':rv_gcs_mid, 'rv_gcs_basal':rv_gcs_basal, 'rv_gcs':rv_gcs, 
                    'rvs_gcs_apex':rvs_gcs_apex, 'rvs_gcs_mid':rvs_gcs_mid, 'rvs_gcs_basal':rvs_gcs_basal, 'rvs_gcs':rvs_gcs}
 
        
        except:
            pass


if __name__ == '__main__':

    startLDT = time.time()

    path_to_models = '/home/ldt18/Desktop/BioBank_Aug22/ALL'
    output_file = './results/global_circum_strain.csv'
    
    fieldnames = ['name','lv_gcs_apex','lv_gcs_mid','lv_gcs_basal','lv_gcs','rv_gcs_apex',
                'rv_gcs_mid','rv_gcs_basal', 'rv_gcs', 'rvs_gcs_apex', 
                'rvs_gcs_mid', 'rvs_gcs_basal', 'rvs_gcs']

    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames= fieldnames)
        writer.writeheader() 

    # listdir in path_model   
    folders = sorted(list(os.listdir(path_to_models)))
    print(len(folders))

    # use the following if you need to visualise one patient. 
    # Change the number in squared brackets depending on which patient you want to visualise.
    # also set plot_figure = True
    #result_1case = find_strain(os.path.join(path_to_models, folders[1]), output_file)

    # use the following line if you want the final excel file for all patients
    all_results = [GC_strain(os.path.join(path_to_models,patient), output_file) for patient in folders]
    #if os.path.isdir(os.path.join(path_to_models,patient)) == True]

    print('TOTAL TIME: ', time.time()-startLDT)


