
'''
15/09/2022 - Laura Dal Toso
Based on A.M's scripts.
Script for the measurement of LV and LV mass and volume from biventricular models.
'''

import os
import numpy as np
import csv
import sys 
sys.path.append('./mesh_tools/')
sys.path.append('../BiV_Modelling_v2/')
from mesh_tools import *


def find_volume( folder,output_file):
    '''
    Inputs:  folder = path to fitted model in .txt format
            ouput_file = path to the output csv file

    Output: csv file containing masses and volumes
    '''

    case =  os.path.basename(os.path.normpath(folder))
    print('case: ', case )
    try:
        input_path_model = '../BiV_Modelling_v2/model/'
        subdivision_matrix_file = os.path.join(input_path_model,
                                                    "subdivision_matrix.txt")
        elements_file = os.path.join(input_path_model,'ETIndicesSorted.txt')
        material_file = os.path.join(input_path_model,'ETIndicesMaterials.txt')

        if not os.path.exists(subdivision_matrix_file):
            ValueError('Missing subdivision_matrix.txt')
        if not os.path.exists(elements_file):
            ValueError('Missing ETIndicesSorted.txt')
        if not os.path.exists(material_file):
            ValueError('Missing ETIndicesMaterials.txt')


        ################## read model ########################
        subdivision_matrix = (np.loadtxt(subdivision_matrix_file)).astype(float)
        faces = np.loadtxt(elements_file).astype(int)-1
        mat = np.loadtxt(material_file, dtype='str')

        # A.M. :there is a gap between septum surface and the epicardial
        #   Which needs to be closed if the RV/LV epicardial volume is needed
        #   this gap can be closed by using the et_thru_wall facets
        et_thru_wall = np.loadtxt(os.path.join(input_path_model,'epi_to_septum_ETindices.txt'),
                                    delimiter='\t').astype(int)-1

        ## convert labels to integer correspondig to the sorted list
        # of unique labels types
        unique_material = np.unique(mat[:,1])
        materials  = np.zeros(mat.shape)
        for index,m in enumerate(unique_material):
            face_index = mat[:,1] == m
            materials[face_index,0] = mat[face_index,0].astype(int)
            materials[face_index, 1] = [index] *np.sum(face_index)

        # add material for the new facets
        new_elem_mat = [list(range(materials.shape[0], materials.shape[0]+et_thru_wall.shape[0])),
            [len(unique_material)]*len(et_thru_wall)]

        # find models at ES and ED
        models  = [k for k in np.sort(list(os.listdir(folder))) if 'Model' in k]
        
        #build dict containing all required measurements for each case
        results_dict = {'case': case}

        if len(models)>0:

                for frame in models: 
                    control_points = np.loadtxt(os.path.join(folder, frame), 
                        delimiter=',',skiprows=1, usecols=[0,1,2]).astype(float)

                    vertices =  np.dot(subdivision_matrix, control_points)
                    faces = np.concatenate((faces.astype(int), et_thru_wall))
                    materials = np.concatenate((materials.T, new_elem_mat), axis = 1).T.astype(int)

                    model = Mesh('mesh')
                    model.set_nodes(vertices*10)
                    model.set_elements(faces)
                    model.set_materials(materials[:,0], materials[:,1])

                    #components list, used to get the correct mesh components: 
                    #['0 AORTA_VALVE' '1 AORTA_VALVE_CUT' '2 LV_ENDOCARDIAL' '3 LV_EPICARDIAL'
                    #' 4 MITRAL_VALVE' '5 MITRAL_VALVE_CUT' '6 PULMONARY_VALVE' '7 PULMONARY_VALVE_CUT'
                    #'8 RV_EPICARDIAL' '9 RV_FREEWALL' '10 RV_SEPTUM' '11 TRICUSPID_VALVE'
                    #'12 TRICUSPID_VALVE_CUT']

                    # Select RV endocardial
                    RV_ENDO = model.get_mesh_component([6,9,10,11], reindex_nodes=False)
                    RV_ENDO.materials = np.zeros_like(RV_ENDO.materials)
                    RV_ENDO.elements[RV_ENDO.materials == 10,:] = \
                        np.array([RV_ENDO.elements[RV_ENDO.materials == 10,0],
                                RV_ENDO.elements[RV_ENDO.materials == 10,2],
                                RV_ENDO.elements[RV_ENDO.materials == 10,1]]).T

                    RV_SEPTUM = model.get_mesh_component([10], reindex_nodes=False)
                    RV_SEPTUM.materials = np.zeros_like(RV_SEPTUM.materials)
                    LV_ENDO = model.get_mesh_component([0,2,4], reindex_nodes=False)
                    LV_ENDO.materials = np.zeros_like(LV_ENDO.materials)

                    RV_EPI = model.get_mesh_component([6,7,8,10,11,12,13] ) # [6,7,8,10,11,12,13] [6,8,10,11,13]
                    RV_EPI.materials = np.zeros_like(RV_EPI.materials)
                    RV_EPI.elements[RV_EPI.materials == 10,:] = \
                        np.array([RV_EPI.elements[RV_EPI.materials == 10,0],
                                RV_EPI.elements[RV_EPI.materials == 10,2],
                                RV_EPI.elements[RV_EPI.materials == 10,1]]).T

                    LV_EPI = model.get_mesh_component([0,1,3,4,5,10,13]) #[0,1,3,4,5,10,13] [0,3,4,10,13]
                    LV_EPI.materials = np.zeros_like(LV_EPI.materials)
                    LV_EPI.elements[LV_EPI.materials == 13,:] = \
                        np.array([LV_EPI.elements[LV_EPI.materials == 13,0],
                                LV_EPI.elements[LV_EPI.materials == 13,2],
                                LV_EPI.elements[LV_EPI.materials == 13,1]]).T
                    
                    # assig a label to each available frame
                    if 10<int(frame[-7:-4])<40:
                        frame = 'es'
                    else:
                        frame = 'ed'

                    lv_endo_vol = LV_ENDO.get_volume() /1000 # Volume in ml
                    rv_endo_vol = RV_ENDO.get_volume() /1000
                    lv_epi_vol = LV_EPI.get_volume() /1000
                    rv_epi_vol = RV_EPI.get_volume() /1000

                    RV_mass = (rv_epi_vol- rv_endo_vol)*1.05
                    LV_mass = (lv_epi_vol- lv_endo_vol)*1.05

                    #assign values to dict
                    results_dict['lv_endo_'+frame] = lv_endo_vol
                    results_dict['rv_endo_'+frame] = rv_endo_vol
                    results_dict['lv_epi_'+frame] = lv_epi_vol
                    results_dict['rv_epi_'+frame] = rv_epi_vol
                    results_dict['lv_mass_'+frame] = LV_mass
                    results_dict['rv_mass_'+frame] = RV_mass

                with open(output_file, 'a',newline="") as f:
                    # print out measurements in spreadsheet
                    writer = csv.writer(f)
                    writer.writerow([case, results_dict['lv_endo_ed'], results_dict['lv_mass_ed'],
                                    results_dict['rv_endo_ed'], results_dict['rv_mass_ed'],
                                    results_dict['lv_endo_es'], results_dict['lv_mass_es'], 
                                    results_dict['rv_endo_es'], results_dict['rv_mass_es']])
                
                return {'case':case, 'lv_endo_ed':results_dict['lv_endo_ed'], 'lv_mass_ed':results_dict['lv_mass_ed'],
                                    'rv_endo_ed':results_dict['rv_endo_ed'], 'rv_mass_ed':results_dict['rv_mass_ed'],
                                    'lv_endo_es':results_dict['lv_endo_es'], 'lv_mass_es':results_dict['lv_mass_es'], 
                                    'rv_endo_es':results_dict['rv_endo_es'], 'rv_mass_es':results_dict['rv_mass_es']}
    except:
        with open(output_file, 'a',newline="") as f:
            # print out measurements in spreadsheet
            writer = csv.writer(f)
            writer.writerow([case, '', '', '',  '', '',  '', '', ''])            


if __name__ == '__main__':

    path_to_models = 'C:/Users/ldt18/Desktop/BBK_results/results_ALL'
    output_file = './results/MassVolumes.csv'
    
    # header of the output spreadsheet. Check that it matches line 145.
    fieldnames = ['name', 'lv_ed', 'lvm_ed', 'rv_ed', 'rvm_ed', 'lv_es', 
            'lvm_es', 'rv_es', 'rvm_es']

    # file containing Hao's mass and volume measurements
    file_Hao = 'C:/Users/ldt18/Desktop/test_cases_edes_volume_mass.csv'
    patients_Hao = []

    with open(file_Hao , 'r') as f:
        Lines = f.readlines()
        list_lines = []
        for i, line in enumerate(Lines):
            list_lines.append(line.strip())

    for i in list_lines[1:]: # skip header
        i = i.split(',')
        patients_Hao.append(i[1])

    with open(output_file, 'w') as f:
        # create output file and write header
        writer = csv.DictWriter(f, fieldnames= fieldnames)
        writer.writeheader() 

    folders = sorted(list(os.listdir(path_to_models)))
    print(len(patients_Hao))

    all_results = [find_volume(os.path.join(path_to_models, patient), output_file) for patient in patients_Hao 
        if patient in folders]
