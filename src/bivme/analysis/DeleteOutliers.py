'''
Laura Dal Toso, 14 September 2022

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Load the two csv files where strains are stored
Strains_csv = 'C:/Users/ldt18/Desktop/forLaura_strain_demographic_wPCs.csv'

df_strains = pd.read_csv(Strains_csv,header=0)


def remove_outliers(df_in, col_name):

    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    #fence_low  = q1-5*iqr
    fence_high = q3+ 5*iqr
    #df_out = df_in.loc[(df_in[col_name] < fence_high)]
    
    df_in.loc[df_in[col_name] > fence_high, col_name] = ' '

    df_out = df_in[col_name]
    #print('\t outliers', df_in[col_name] > fence_high)

    return df_out

new_cols = {}
for i, element in enumerate(df_strains.columns):
    if 1<=i<=18:
        new_cols[element] = remove_outliers(df_strains, str(element))
    else:
        
        new_cols[element] = df_strains[element] 

print(new_cols['CC_manual_apex'] )
new_df = pd.DataFrame(new_cols)

new_df.to_csv('strain_demographic_wPCs_noOutliers.csv',index=False)