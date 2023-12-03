import csv
import sys
import pandas as pd
import itertools
import more_itertools as mit

from bivme.fitting import *
from bivme.preprocessing import Contours as cont
from bivme.preprocessing.cvi42.CVI42XML import *


fieldnames = ['patient', 'frames', 'MITRAL_VALVE', 'TRICUSPID_VALVE', 'AORTA_VALVE', 'APEX_POINT']


def ReformatFiles(folder, gpfile, sliceinfofile, **kwargs):
    '''
    Author : Laura Dal Toso
    Date: 20/20/2022
    Based on: R.B's script extract_gp_points_kcl_RB.py

    -----------------------------------------
    This function can change some labels from cvi42 format to the format required 
    by the biVfitting code. 
    It extracts the apex, mitral, aorta and tricuspid valve points, which may be 
    labelled as 'LV/RV_EXTENT'. 

    This function also pre-processes the SliceInfo file so that only the information 
    matching the points in the GPFire is stored. 

    ----------------------------------------
    Input: 
        - folder where the GPFile and SliceInfoFile are stored
        - gpfile: name of the GPFile   (i.e. 'GPFile.txt')
        - sliceinfofile: name of the slice info file (i.e 'SliceInfoFile.txt') 

    Output: 
        -  GPFIle_proc : processed GPFile 
        - SliceInfoFile_proc: processed SliceInfoFile
    '''

    case =  os.path.basename(os.path.normpath(folder))
    print('case: ', case )

    # define path to input GPfile and SliceInfoFile
    contour_file = os.path.join(folder, gpfile) 
    metadata_file = os.path.join(folder, sliceinfofile)

    contours  = cont.Contours()
    contours.read_gp_files(contour_file,metadata_file)

    case =  os.path.basename(os.path.normpath(folder))

    all_frames = pd.read_csv(contour_file, sep = '\t') 
    time_frames = sorted(np.unique([i[6] for i in all_frames.values]))  # this is the range of time frame numbers in the GPFiles
    
    try:
        contours.find_timeframe_septum()
    except:
        err = 'Computing septum'
        print('Fail',err)
        #print('\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail', err))

    try:
        contours.find_timeframe_septum_inserts(time_frame=time_frames)
    except:
        err = 'Computing inserts'
        print( 'Fail',err)
        #print('\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail',err))
    
    try:
        contours.find_apex_landmark(time_frame=time_frames)
    except:
        err = 'Computing apex'
        print('\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail',err))

    try:
        #contours.find_timeframe_valve_landmarks()
        if 'LAX_LV_EXTENT' in contours.points.keys():
            for index,point in enumerate(contours.get_timeframe_points(
                                'LAX_LV_EXTENT', time_frames)[1]):
                # the extents has 3 points, for each extent we need to
                # select the first 2 corresponding to the valve
                # the output from get_timeframe_points is already sorted by timeframe
                # therefor we pick the firs to points by timeframe

                # In this dataset the LAX_EXTENT in 3CH is not corresponding to
                # the mitral valve so we need to exclude them
                # if there are aorta points on the same timeframe
                # then is a 3ch and we need to exclude them
                aorta_points,_ = contours.get_frame_points('AORTA_VALVE',
                                                        point.sop_instance_uid)
                atrial_extend,_ = contours.get_frame_points('LAX_LA_EXTENT',
                                                            point.sop_instance_uid)
                if len(aorta_points)>0:
                    continue
                if len(atrial_extend)>0:
                    continue
                if (index+1) % 3 !=0:
                    contours.add_point('MITRAL_VALVE',point)
            del contours.points['LAX_LV_EXTENT']

        if 'LAX_LA_EXTENT' in contours.points.keys():
            for index, point in enumerate(contours.get_timeframe_points(
                'LAX_LA_EXTENT', time_frames)[1]):
                if(index +1)%3 !=0:
                    contours.add_point('MITRAL_VALVE', point)
            del contours.points['LAX_LA_EXTENT']

        if 'LAX_RV_EXTENT' in contours.points.keys():
            for index,point in enumerate(contours.get_timeframe_points(
                    'LAX_RV_EXTENT', time_frames)[1]):
                if (index + 1) % 3 != 0:
                    contours.add_point('TRICUSPID_VALVE', point)
            del contours.points['LAX_RV_EXTENT']
    except:
            err = 'Computing valve landmarks'
            print(case, 'Fail',err)
            #print( '\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail',err))

    cvi_cont = CVI42XML()
    cvi_cont.contour = contours

    new_gpfilename = 'GPFile_proc.txt'
    new_sliceinfofilename = 'SliceInfo_proc.txt'

    output_gpfile = os.path.join(folder,new_gpfilename)
    output_metafile = os.path.join(folder, new_sliceinfofilename)

    cvi_cont.export_contour_points(output_gpfile)
    cvi_cont.export_dicom_metadata(output_metafile)

    return new_gpfilename, new_sliceinfofilename


def Landmarks_Dict(data_set, out_file, case):
    '''
    Author: Laura Dal Toso
    Date: 25/07/22

    ----------------------------------------------------------
    Extracts the landmark points from the GPFiles and stores how many landmarks
    there are for each contour type. This function also interpolates landmark positions
    if they are missing from maximum two consecutive frames. If landmarks are missing 
    from more than 2 consecutive frames, teh case is discarded. 

    **note: Before using, check that exp_values_dict contains the correct expected number for each 
            type of landmarks. For example, if you expect at least N mitral valve points in 
            your dataset, set exp_values_dict = {'MITRAL_VALVE': (N, col3)}.
    ----------------------------------------------------------

    Input: 
        - data_set: a list of tuples with structure: (frame_num, GPDataset instance)
        - out_file: path to output file (.csv)
        - case: patient name

    Output: 
        - dataframe dictionary containing all landmark points all all frames, 
            together with a status label (changes, unchanges, deleted..)

    '''
    
    dataframe_dict = {'patient': [case], 'frames': [], 'MITRAL_VALVE':[],  
                        'TRICUSPID_VALVE':[],'AORTA_VALVE':[],'APEX_POINT':[], 'status' : [] }

    for idx, item in enumerate(data_set):
            frame_num = item[0] # frame number
            data_set = item[1]  # GPDataset instance at given frame
            # load 3D coordinates for each landmark type
            mitral_points = data_set.points_coordinates[
                data_set.contour_type == ContourType.MITRAL_VALVE]
            aorta_points = data_set.points_coordinates[
                data_set.contour_type == ContourType.AORTA_VALVE]
            tricuspid_points = data_set.points_coordinates[
                data_set.contour_type == ContourType.TRICUSPID_VALVE]
            apex_point = data_set.points_coordinates[
                data_set.contour_type == ContourType.APEX_POINT]    

            # save 3D coordinates at each frame for each landmark
            #dataframe_dict['patient'].append('')    
            dataframe_dict['frames'].append(frame_num)            
            dataframe_dict['MITRAL_VALVE'].append(mitral_points)
            dataframe_dict['TRICUSPID_VALVE'].append(tricuspid_points)
            dataframe_dict['AORTA_VALVE'].append(aorta_points)
            dataframe_dict['APEX_POINT'].append(apex_point)

    # count how many landmarks of each type are stored in GPFiles
    col1 = [k for k in dataframe_dict[list(fieldnames)[0]]] #patient
    col2 = [k for k in dataframe_dict[list(fieldnames)[1]]] #frames
    col3 = [len(k) for k in dataframe_dict[list(fieldnames)[2]]] #mitral
    col4 = [len(k) for k in dataframe_dict[list(fieldnames)[3]]] #tricuspid
    col5 = [len(k) for k in dataframe_dict[list(fieldnames)[4]]] #aorta
    col6 = [len(k) for k in dataframe_dict[list(fieldnames)[5]]] #apex 
	

    with open(out_file, 'a') as f:
        df = pd.DataFrame(list(itertools.zip_longest(*[col1, col2, col3, col4, col5, col6])))
        df.to_csv(f, index=False, line_terminator='\n', header=False)

    # set expected umber of landmarks for each landmark type
    exp_values_dict = {'MITRAL_VALVE': (6, col3), 'TRICUSPID_VALVE':(2, col4), 
                        'AORTA_VALVE':(2,col5) , 'APEX_POINT':(1, col6)}
    
    status = ['', '']
    for label, items in exp_values_dict.items():
        exp_num = items[0]  # ecpected number of landmarks
        col = items[1]      # number of landmarks at each frame
        
        # look for missing landmarks
        if any(t < exp_num  for t in col):

            masked_col = np.ma.getmask(np.ma.masked_less(col, exp_num)) 
            data = np.where(masked_col == True)[0]
            # count how many consecutive frames have missing landmarks 
            count_dups = [len(list(group)) for group in mit.consecutive_groups(data)]
            # if up to 2 frames have missing landmarks, interpolate the values
            
            if any(t > 2 for t in count_dups):
                # delete dataset if landmarks are missing in 3 or more consecutive frames
                dataframe_dict = {'patient': case, 'status': 'deleted'}
                with open(out_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow('deleted')
                return dataframe_dict
            
            elif any(t <= 2 for t in count_dups): 
                index = [i for (i, item) in enumerate(col) if item <exp_num]

                for idx in index:
                    try:
                        if col[idx-1] !=0 and col[idx+1] !=0:
                            avg_val = [(g + h) / 2 for g, h in zip(
                                dataframe_dict[label][idx-1], dataframe_dict[label][idx+1])]
                            dataframe_dict[label][idx] = np.array(avg_val)
                            
                        elif col[idx-1] !=0:
                            dataframe_dict[label][idx] = dataframe_dict[label][idx-1]

                        elif col[idx+1] !=0:
                            dataframe_dict[label][idx] = dataframe_dict[label][idx+1]
                    except:
                        print('ERROR in dictionary creation') 
                        ValueError
       
                    dataframe_dict['status'].append('changed')
                status.append('changed')
                    
        else:
            dataframe_dict['status'].append('unchanged')
            status.append('unchanged')

            continue  

    with open(out_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(status)

    return dataframe_dict


def tracking_landmarks(folder, gpfilename, SliceInfoname, **kwargs):
    '''
    Author: ldt
    Date: 25/07/22
    ----------------------
    This function checks that all necessary landmarks are present in the input gpfile (by calling Landmarks_dict) 
    and generates a new dataset with the same structure as gpdata, that can be used for further processing. 
        
    ----------------------------------------------------------

    Input: 
        - folder where the GPFile and SliceInfoFile are stored
        - gpfilename: name of the GPFile   (i.e. 'GPFile.txt')
        - SliceInfoname: name of the slice info file (i.e 'SliceInfoFile.txt') 
    Output: 
         - dataframe with GPData structure, where missing landmarks have been replaced

    '''
    try:
        if 'output_csv' in kwargs:
            landmarks_csv = kwargs.get('output_csv', None)

        final_status = 'unchanged'
        filename = os.path.join(folder, gpfilename)
        filenameInfo = os.path.join(folder, SliceInfoname)

        # chose which frames to upload from the GPFile
        all_frames = pd.read_csv(filename, sep = '\t') 
        frames_to_fit = sorted(np.unique([i[6] for i in all_frames.values]))
        case = os.path.basename(os.path.normpath(folder))
        
        # build a list containing GPData structures, one for each frame
        print('-----> Uploading GP dataset')
        data_set = [] # structure (frame, GPData)
        for num in frames_to_fit:
            data_set.append((num, GPDataSet(filename,filenameInfo, case, sampling = 1, time_frame_number = num)))
        
        # Find missing landmarks and interpolate if necessary
        print('-----> Creating Dataframe')
        dataframe_dict = Landmarks_Dict(data_set, landmarks_csv, case)

        # find which frames have been changed by Landmarks_Dict()
        try: 
            status = dataframe_dict['status']
            contour_idx = [i for i,k in enumerate(status) if k =='changed']
            if len(contour_idx)>0:
                    final_status = 'changed'
        except:   
            pass

        try: 
            status = dataframe_dict['status']
            if status=='deleted':
                final_status = 'deleted'
        except:   
            pass
    
        # write new GPFile 
        dict_contypes = {'MITRAL_VALVE': ContourType.MITRAL_VALVE, 'TRICUSPID_VALVE':ContourType.TRICUSPID_VALVE,
                            'AORTA_VALVE': ContourType.AORTA_VALVE,'APEX_POINT': ContourType.APEX_POINT}

        tracked_dataset = []
        for i,data in enumerate(data_set): # this is iterating over time frames
            gpdata = data[1]    
            if final_status == 'deleted':
                continue
            elif final_status == 'changed':
                for contour in contour_idx:
                    cont_name = fieldnames[contour+2]
                    newpoints = dataframe_dict[cont_name][i]
                    #assign the new points to the datafile 
                    try:
                        gpdata.points_coordinates[gpdata.contour_type== dict_contypes[cont_name]] = newpoints
                    except:
                        slice_number = np.unique(data_set[0][1].slice_number[data_set[0][1].contour_type ==dict_contypes[cont_name]])
                        for point in newpoints:
                            gpdata.add_data_points(np.expand_dims(point, axis=0), [dict_contypes[cont_name]], slice_number, [1] )
                    tracked_dataset.append((data[0] , gpdata))
            else:    
                tracked_dataset.append(data)
                # clean points between the tricuspid and mitral valves
                #clean 3ch view (delete it because some 3ch points should be labelled as septum but they are not)

        return tracked_dataset

    except:
        return 'ERROR'
    
def Clean_contours(folder, tracked_dataset, gpfilename_out, **kwargs):
    '''
    Author: Laura Dal Toso
    Date: 20/10/2022
    --------------------------------------------------------------
    This function deletes unwanted points from GPdata structures. 
        
    -------------------------------------------------------------

    Input: 
        - folder where the GPFile and SliceInfoFile are stored
        - tracked_dataset: GPData structure 
        - gpfilename_out: name of output GPFile 

    Output: 
         - GPFile.txt where unwanted points have been deleted

    '''    
    with open(os.path.join(folder, gpfilename_out), 'w') as f:
        f.write('{0:}\t{1}\t{2}\t'.format('x', 'y', 'z')+ '{0}\t'.format('contour type')
            + '{0}\t{1}\t{2}'.format('sliceID','weight','time frame')+'\n')   
    
    clean_dataset = []
    for i,data in enumerate(tracked_dataset):   
        gpdata = data[1]           
        gpdata.clean_LAX_contour()
        gpdata.write_gpfile(os.path.join(folder, gpfilename_out), time_frame = data[0])
        clean_dataset.append((data[0] , gpdata))

    return clean_dataset


def findED_ESframe(case, data_set, **kwargs):   
    '''
    Author: Laura Dal Toso
    Date: 20/10/2022
    --------------------------------------------------------------
    This function finds where the ES frame is located, based on the distance between the apex and the aorta valves.
    -------------------------------------------------------------

    Input: 
        - folder where the GPFile and SliceInfoFile are stored
        - data_set: gpdata structure containing the guide points (from the GPFile)

    Output: 
         - spreadsheet containing patient name, ED frame, ES frame

    '''  

    if 'case_frame_dict' in kwargs:
        case_frame_dict = kwargs.get('cases_frame_dict', None)
    else: 
        case_frame_dict = {'case':[]}


    dist_aorta_apex = []
    for data in data_set:
        gpdata = data[1] 
        dist_aorta_apex.append(gpdata.dist_mitral_to_apex())

    if None in dist_aorta_apex:
        ED_frame = None
        ES_frame = None

    elif len(dist_aorta_apex) >0:
        mindist_idx = np.argmin(dist_aorta_apex)
        ED_frame = 1
        ES_frame = data_set[mindist_idx][0]
    
    with open('./results/case_id_frame.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow((case, ED_frame, ES_frame))   


