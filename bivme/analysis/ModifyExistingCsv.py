


import numpy as np
import pandas as pd
import numpy as np
import os 
import sys 
sys.path.append('../BiV_Modelling_v2/')
sys.path.append('./mesh_tools/')
from mesh_tools import *
from Mass_volume import find_volume
from compute_circum_global_strain import GC_strain
from compute_global_longitudinal_strain import GL_strain
from Mass_volume import find_volume

Big_csv = './forLaura_strain_demographic.csv'

#read info from csv file containing tagging strain results
df = pd.read_csv(Big_csv,header=0)
patients = df.loc[:, df.columns[0]]


'''
1 lv_gls_4CH 
2 lv_gls_3CH 	
3 rv_gls_4CH 
4 rv_gls_3CH 
5 rvs_gls_4CH	
6 rvs_gls_3CH	

7 lv_gcs_apex	
8 lv_gcs_mid	
9 lv_gcs_basal	
10 rv_gcs_apex	
11 rv_gcs_mid	
12 rv_gcs_basal	
13 rvs_gcs_apex	
14 rvs_gcs_mid	
15 vs_gcs_basal	

CC_manual_apex	
CC_manual_middle	
CC_manual_base	

19 lv_ed	
20 lvm_ed	
21 rv_ed	
22 rvm_ed	
23 lv_es	
24 lvm_es	
25 rv_es	
26 rvm_es
'''

#select unique patient IDs
unique_ids = list(np.unique(patients))

path_to_models = 'C:/Users/ldt18/Desktop/BBK_results/results_ALL'

with open('list_batch2.txt ', 'r') as f:
    Lines = f.readlines()
    batch2list = []
    for i, line in enumerate(Lines):
        batch2list.append(line.strip())


for patient in batch2list[1:]:
        try:
            i = unique_ids.index(int(patient))

            print(patient, i)

            folder = os.path.join(path_to_models, str(patient))
            output_file = './strain_AY/test.csv'

            GC_strain_dict = GC_strain(folder,output_file)
            GL_strain_dict = GL_strain(folder,output_file)

            df.at[i,'lv_gls_4CH']= GL_strain_dict['lv_gls_4CH']
            df.at[i,'lv_gls_3CH']= GL_strain_dict['lv_gls_3CH']
            df.at[i,'rv_gls_4CH']= GL_strain_dict['rv_gls_4CH']
            df.at[i,'rv_gls_3CH']= GL_strain_dict['rv_gls_3CH']
            df.at[i,'rvs_gls_4CH']= GL_strain_dict['rvs_gls_4CH']
            df.at[i,'rvs_gls_3CH']= GL_strain_dict['rvs_gls_3CH']

            df.at[i,'lv_gcs_apex']= GC_strain_dict['lv_gcs_apex']
            df.at[i,'lv_gcs_mid']= GC_strain_dict['lv_gcs_mid']
            df.at[i,'lv_gcs_basal']= GC_strain_dict['lv_gcs_basal']
            df.at[i,'rv_gcs_apex' ]= GC_strain_dict['lv_gcs_basal']
            df.at[i,'rv_gcs_mid']= GC_strain_dict['rv_gcs_apex']
            df.at[i,'rv_gcs_basal']= GC_strain_dict['rv_gcs_basal']
            df.at[i,'rvs_gcs_apex']= GC_strain_dict['rvs_gcs_apex']
            df.at[i,'rvs_gcs_mid']= GC_strain_dict['rvs_gcs_mid']
            df.at[i,'rvs_gcs_basal']= GC_strain_dict['rvs_gcs_basal']

            MassVol_dict = find_volume( folder,output_file)
            df.at[i,'lv_ed']= MassVol_dict['lv_endo_ed']
            df.at[i,'lvm_ed']= MassVol_dict['lv_mass_ed']
            df.at[i,'rv_ed']= MassVol_dict['rv_endo_ed']
            df.at[i,'rvm_ed']= MassVol_dict['rv_mass_ed']
            df.at[i,'lv_es']= MassVol_dict['lv_endo_es']
            df.at[i,'lvm_es']= MassVol_dict['lv_mass_es']
            df.at[i,'rv_es']= MassVol_dict['rv_endo_es']
            df.at[i,'rvm_es']= MassVol_dict['rv_mass_es']
        
        except KeyboardInterrupt:
            break
        except:
            print('dropped case:', patient)
            df.drop(labels=i, axis=0)



df.to_csv('./results/all_stats44K.csv', mode='w', index = False, header=True)
