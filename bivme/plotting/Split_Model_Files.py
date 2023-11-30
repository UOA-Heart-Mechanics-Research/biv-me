# Created by LDT on 26 April 2022
# This script contains functions to split a Model.txt file containing all frames into many files, each for one frame.

import os
import pandas as pd 


main_path = 'C:/Users/ldt18/Desktop/Dev_BioBank'
cases_folder = os.path.join(main_path, './results')
cases_list = [os.path.join(cases_folder, batch) for batch in os.listdir(cases_folder) ]

   
for folder in cases_list: 
    case =  os.path.basename(os.path.normpath(folder))
    Model= os.path.join(main_path, folder+'/'+case+'_Model_file.txt')

    print(Model)

    df =  pd.read_table(open(Model), sep='\t', header = 0)

    frame = []
    frame_num = [900]
    for line_index,line in enumerate(df.values):

        frame = list(map(float,line[0].split()))
        num = frame[-1]
        df = pd.DataFrame({'x': [float(frame[0])], 'y': [float(frame[1])], 'z':[float(frame[2])], 'frame':[int(frame[3])]})

        if num!=frame_num[-1]:
            DataFile =  os.path.join(main_path, folder+'/'+case+'_Model_Frame_'+ format(int(num), "03")+'.txt')
            #DataFile.touch(exist_ok=True)

            with open(DataFile, 'w', newline='') as file:
                    file.write(df.to_csv(header=True, index=False,sep=','))

        elif num == frame_num[-1]:
            DataFile =  os.path.join(main_path, folder+'/'+case+'_Model_Frame_'+ format(int(num), "03")+'.txt')

            with open(DataFile, 'a', newline='') as file:
                    file.write(df.to_csv(header=False, index=False,sep=','))
        
        frame_num.append(num)

    #df = df.replace('nan', 0)


    #df = df.assign(**{'Frame': df['Frame'].fillna(df['number'])})   

    #df.assign(**{'Frame': df['Frame'].mask(lambda x: x == 'nan', df['number'])})
    #df['new_column'] = df['Frame'].astype(str) + df['number']
 
    #print(np.unique(df['Frame']))
    #print(np.unique(df['number']))

    #pd.DataFrame(df.T[0].values.reshape(df.shape[1]//2,2))
