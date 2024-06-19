import csv
import pandas as pd
import itertools
import more_itertools as mit
import copy
import math
from pydicom.filereader import dcmread

from bivme.fitting import *
from bivme.preprocessing import Contours as cont
from bivme.preprocessing.cvi42.CVI42XML import *
from bivme.preprocessing.read_cim_guide_points import read_guidepoints, read_offsets


fieldnames = [
    "patient",
    "frames",
    "MITRAL_VALVE",
    "TRICUSPID_VALVE",
    "AORTA_VALVE",
    "APEX_POINT",
]


def ReformatFiles(folder, gpfile, sliceinfofile, temporal_matching=False, **kwargs):
    """
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
    """

    case = os.path.basename(os.path.normpath(folder))
    print("case: ", case)

    # define path to input GPfile and SliceInfoFile
    contour_file = os.path.join(folder, gpfile)
    metadata_file = os.path.join(folder, sliceinfofile)
    contours = cont.Contours()

    print("Reading GPFile and SliceInfoFile... ", end="")
    contours.read_gp_files(contour_file, metadata_file)
    print("done")

    case = os.path.basename(os.path.normpath(folder))

    all_frames = pd.read_csv(contour_file, sep="\t")
    time_frames = sorted(
        np.unique([i[6] for i in all_frames.values])
    )  # this is the range of time frame numbers in the GPFiles

    try:
        contours.find_timeframe_septum()
    except:
        err = "Computing septum"
        print("Fail", err)
        # print('\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail', err))

    try:
        contours.find_timeframe_septum_inserts(time_frame=time_frames)
    except:
        err = "Computing inserts"
        print("Fail", err)
        # print('\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail',err))

    try:
        contours.find_apex_landmark(time_frame=time_frames)
    except:
        err = "Computing apex"
        print("Fail", err)
        # print('\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail',err))

    try:
        contours.find_pulmonary_valve_landmarks(timeframe=time_frames)
    except:
        err = "Computing pulmonary valve"
        print("Fail", err)
        # print('\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail',err))

    try:
        # contours.find_timeframe_valve_landmarks()
        if "LAX_LV_EXTENT" in contours.points.keys():
            for index, point in enumerate(
                contours.get_timeframe_points("LAX_LV_EXTENT", time_frames)[1]
            ):
                # Joshua Dillon (15/04/2024) 
                # Commented out code below that was extracting mitral valve from LAX_LV_EXTENT. Choosing to extract from LAX_LA_EXTENT instead
                pass

                # the extents has 3 points, for each extent we need to
                # select the first 2 corresponding to the valve
                # the output from get_timeframe_points is already sorted by timeframe
                # therefor we pick the firs to points by timeframe

                # In this dataset the LAX_EXTENT in 3CH is not corresponding to
                # the mitral valve so we need to exclude them
                # if there are aorta points on the same timeframe
                # then is a 3ch and we need to exclude them
                # aorta_points, _ = contours.get_frame_points(
                #     "AORTA_VALVE", point.sop_instance_uid
                # )
                # atrial_extend, _ = contours.get_frame_points(
                #     "LAX_LA_EXTENT", point.sop_instance_uid
                # )
                # if len(aorta_points) > 0:
                #     pass
                # if len(atrial_extend) > 0:
                #     pass
                # # if (index + 1) % 3 != 0:
                # #     contours.add_point("MITRAL_VALVE", point)
            del contours.points["LAX_LV_EXTENT"]

        if "LAX_LA_EXTENT" in contours.points.keys():
            for index, point in enumerate(
                contours.get_timeframe_points("LAX_LA_EXTENT", time_frames)[1]
            ):
                # Joshua Dillon (15/04/2024) 
                # Extracting mitral valve from LAX_LA_EXTENT instead of LAX_LV_EXTENT. 
                # Always the first two points
                if (index + 1) % 3 != 0:
                    contours.add_point("MITRAL_VALVE", point)
            del contours.points["LAX_LA_EXTENT"]

        if "LAX_RV_EXTENT" in contours.points.keys():
            for index, point in enumerate(
                contours.get_timeframe_points("LAX_RV_EXTENT", time_frames)[1]
            ):
                # Joshua Dillon (15/04/2024) 
                # Commented out code below that was extracting tricuspid valve from LAX_RV_EXTENT. Choosing to extract from LAX_RA_EXTENT instead
                pass
                # if (index + 1) % 3 != 0:
                #     contours.add_point("TRICUSPID_VALVE", point)
            del contours.points["LAX_RV_EXTENT"]

        if "LAX_RA_EXTENT" in contours.points.keys():
            for index, point in enumerate(
                contours.get_timeframe_points("LAX_RA_EXTENT", time_frames)[1]
            ):
                # Joshua Dillon (15/04/2024) 
                # Extracting mitral valve from LAX_RA_EXTENT instead of LAX_RV_EXTENT. 
                # Always the first two points
                if (index + 1) % 3 != 0:
                    contours.add_point("TRICUSPID_VALVE", point)
            del contours.points["LAX_RA_EXTENT"]

    except:
        err = "Computing valve landmarks"
        print(case, "Fail", err)
        # print( '\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail',err))

    if temporal_matching == True:
        contours = Temporal_Matching(contours)

    cvi_cont = CVI42XML()
    cvi_cont.contour = contours

    new_gpfilename = "GPFile_proc.txt"
    new_metadatafilename = "SliceInfoFile_proc.txt"
    output_gpfile = os.path.join(folder, new_gpfilename)
    output_metadatafile = os.path.join(folder, new_metadatafilename)
    cvi_cont.export_contour_points(output_gpfile)
    cvi_cont.export_dicom_metadata(output_metadatafile)

    return output_gpfile, output_metadatafile


def Temporal_Matching(contours):
    """
    Author: Joshua Dillon
    Date: 6/12/2023

    ----------------------------------------------------------
    Resamples contours to the minimum number of frames of any one contour, thereby accounting for
    phase inconsistencies in CMR protocols.

    ----------------------------------------------------------

    Input:
        - contours: Contours instance

    Output:
        - contours_resampled: Contours instance with resampled contours
    """
    all_frames = []
    frame_dict = {}
    for contour_type, points in contours.points.items():
        frames = np.unique([point.time_frame for point in points])
        all_frames.append(len(frames))

        # express frames as a proportion of the total number of frames
        frames_prop = frames / max(frames)
        frame_dict[contour_type] = [frames, frames_prop]

    min_frames = np.arange(0, min(all_frames))
    min_frames_prop = min_frames / max(min_frames)
    contours_resampled = cont.Contours()

    for contour_type, points in contours.points.items():
        frames = frame_dict[contour_type][0]
        frames_prop = frame_dict[contour_type][1]
        if np.all(frames == min_frames):
            contours_resampled.points[contour_type] = points
            for point in points:
                if point.sop_instance_uid not in contours_resampled.frame.keys():
                    contours_resampled.add_frame(
                        point.sop_instance_uid, contours.frame[point.sop_instance_uid]
                    )
        else:
            new_frames = []
            for i in range(0, len(min_frames)):
                min_dist = np.argmin(np.abs(min_frames_prop[i] - frames_prop))
                new_frames.append(frames[min_dist])

            for point in points:
                if point.time_frame not in new_frames:
                    pass
                else:
                    new_point = point.deep_copy_point()
                    new_point.time_frame = min_frames[
                        new_frames.index(point.time_frame)
                    ]
                    if (
                        new_point.sop_instance_uid
                        not in contours_resampled.frame.keys()
                    ):
                        mod_frame = Frame(
                            contours.frame[new_point.sop_instance_uid].image_id,
                            contours.frame[new_point.sop_instance_uid].position,
                            contours.frame[new_point.sop_instance_uid].orientation,
                            contours.frame[new_point.sop_instance_uid].pixel_spacing,
                        )
                        mod_frame.time_frame = new_point.time_frame
                        contours_resampled.add_frame(
                            new_point.sop_instance_uid, mod_frame
                        )
                    contours_resampled.add_point(contour_type, new_point)

    print("Resampled contours to minimum number of frames: ", min(all_frames))
    return contours_resampled


def CIM_Correction(folder, gpfile, sliceinfofile, cim_data, image_ptrs, cim_offsets=None, valve_points=True, slice_corrections=True, **kwargs):
    '''
    Author: Joshua Dillon
    Date: 11/12/2023

    ----------------------------------------------------------
    This function adds valve points generated from CIM RVLV and replaces existing valve points. 
    Additionally, manual slice corrections derived from CIM RVLV are applied to regular guidepoints.
    Process requires matching the slice ID from CIM to CIRCLE, which is done by matching the SOP Instance UID 
    in the DICOM metadata.

    ----------------------------------------------------------
    Input:
        - folder: path to folder containing preprocessed GPFile and SliceInfoFile
        - gpfile: name of GPFile
        - sliceinfofile: name of SliceInfoFile
        - cim_data: path to folder containing CIM RVLV model data
        - image_ptrs: path to imageptrs
        - (OPTIONAL) cim_offsets: path to folder containing slice corrections from CIM RVLV
        - (OPTIONAL) valve_points: boolean indicating whether to add valve points from CIM RVLV
        - (OPTIONAL) slice_corrections: boolean indicating whether to apply slice corrections from CIM RVLV 
    Output:
        - GPFile_CIM : processed GPFile 
        - SliceInfoFile_CIM: processed SliceInfoFile
    '''

    analystID = 'DZ' # TODO make this an input?
    
    case =  os.path.basename(os.path.normpath(folder))
    print('case: ', case )

    # define path to input GPfile and SliceInfoFile
    contour_file = os.path.join(folder, gpfile) 
    metadata_file = os.path.join(folder, sliceinfofile)
    contours  = cont.Contours()
    
    print('Reading GPFile and SliceInfoFile... ', end='')
    contours.read_gp_files(contour_file,metadata_file)
    print('done')

    contours_CIM = copy.deepcopy(contours)
    slices_CIM = {}
    num_frames = max(contours._time_uid_map.keys())+1
    # read in CIM guidepoints
    guidepoints = read_guidepoints(case, analystID, cim_data)
    del guidepoints[0] # first entry is empty
    print('CIM guide points read in successfully')

    ## match slice IDs from CIM to CIRCLE
    # find number of sax slices
    sax_dict={}
    for i in range(0,num_frames):
        sax_slices = [k['slice'] for k in guidepoints if k['series'] == '0' and k['frame'] == str(i)]
        sax_dict.update({i:max(sax_slices)})
    lax_dict={}
    for i in range(0,num_frames):
        lax_slices = [k['slice'] for k in guidepoints if k['series'] == '1' and k['frame'] == str(i)]
        lax_dict.update({i:max(lax_slices)})
    
    sax_num = int(max(sax_dict[0])) + 1
    
    # read in imageptrs file
    imageptrs = os.path.join(cim_data, case,"system",case+".img_imageptr")
    p = open(imageptrs, "r")
    ptrs = p.readlines()
    del ptrs[0]
    for i in ptrs:
        j=i.split('\t')
        j[-1] = str.replace(j[-1], '\n', '')
        
        # replace IMAGEPATH with image directory
        j[-1] = str.replace(j[-1], 'IMAGEPATH\\', image_ptrs)
        ptrs[ptrs.index(i)] = j
    p.close()
    print('Image pointers read in successfully')

    uid_slice_match = {}
    uid_coords_match = {}
    # for each frame, match the cim slice to the circle slice
    for frame in range(0,num_frames):
        for k in ptrs:
            if int(k[2]) == frame:
                # read in dicom header
                dcm = dcmread(k[-1])
                # find the sop UID
                sop = dcm.SOPInstanceUID
                if int(k[0])==0: # if short axis slice (series 0)
                    cim_slice = int(k[1])
                else: # if long axis slice (series 1)
                    cim_slice = int(k[1])+sax_num

                uid_slice_match.update({sop:cim_slice})
                uid_coords_match.update({sop:[dcm.ImagePositionPatient,dcm.ImageOrientationPatient, dcm.PixelSpacing,frame]})

    if slice_corrections == True:
        # read in slice corrections
        offsets = extract_offsets(os.path.join(cim_offsets, case, "OffsetFile.txt"))
        print('Slice corrections read in successfully')

        # apply slice corrections
        for contour_type, points in contours_CIM.points.items():
            for point in points:
                if point.sop_instance_uid in uid_slice_match.keys():
                    # TODO - don't overwrite inbuilt function "slice"!
                    slice = uid_slice_match[point.sop_instance_uid]
                    point.coordinates = point.coordinates + offsets[slice][0]
                    contours_CIM.frame[point.sop_instance_uid].position = contours_CIM.frame[point.sop_instance_uid].position + offsets[slice][0]
                    contours_CIM.frame[point.sop_instance_uid].orientation = contours_CIM.frame[point.sop_instance_uid].orientation + offsets[slice][1]
        print('Slice corrections applied successfully')
    
    if valve_points == True: # TODO: ability to toggle on/off RV inserts?
        valve_types = ['MITRAL_VALVE', 'TRICUSPID_VALVE', 'AORTA_VALVE', 'APEX_POINT', 'RV_INSERT', 'PULMONARY_VALVE'] 
        cim_valve_types = ['GP_VALVE_MITRAL', 'GP_VALVE_TRICUSPID', 'GP_VALVE_AORTIC', 'GP_LV_EPI_APEX', 'GP_RV_INSERTION', 'GP_VALVE_PULMONARY'] 
        cim_warp_types = ['GP_VALVE_MITRAL_WARP', 'GP_VALVE_TRICUSPID_WARP', 'GP_VALVE_AORTIC_WARP', 'GP_LV_EPI_APEX_WARP', 'GP_VALVE_PULMONARY_WARP'] 
        
        # delete existing valve points
        print('Deleting existing valve points...')
        for contour_type, points in contours.points.items():
            if contour_type in valve_types:
                to_del = np.arange(0,len(points))
                contours_CIM.delete_point(contour_type, to_del)
        
        # ensure that sliceinfofile has all available slices
        print('Adding missing slices...')


        # match slices beween cim and circle
        frame_insert_point = len(contours_CIM.frame.keys())
        count=0
        for uid,coords in uid_coords_match.items():
            # if cim has a slice that circle doesn't
            if uid not in contours_CIM.frame.keys():
                count+=1
                new_frame = Frame(image_id= frame_insert_point + count,
                                  position = coords[0], orientation = coords[1], pixel_spacing = coords[2])
                new_frame.time_frame = coords[3]
                contours_CIM.add_frame(uid, new_frame)
        
        for uid,frame in contours_CIM.frame.items():
            if uid not in uid_coords_match.keys():
                # find closest slice by position, orientation
                min_pos_d = np.inf
                min_ori_d = np.inf
                for cim_uid, cim_coords in uid_coords_match.items():
                    pos_d = math.dist(frame.position, cim_coords[0])
                    ori_d = math.dist(frame.orientation, cim_coords[1])
                    if pos_d < min_pos_d:
                        min_pos_d = pos_d
                        closest_pos_uid = cim_uid
                    if ori_d < min_ori_d:
                        min_ori_d = ori_d
                        closest_ori_uid = cim_uid
                # if closest_ori_uid == closest_pos_uid:
                #     uid_slice_match.update({uid:uid_slice_match[closest_ori_uid]})
                #     uid_coords_match.update({uid:uid_coords_match[closest_ori_uid]})

        # add valve points
        print('Adding valve points...')
        
        # get offsets
        offsets = read_offsets(caseID=case, analystID=analystID, rvlvpath=cim_data)
        
        for point in guidepoints:
            if point['name'] in cim_valve_types or point['name'] in cim_warp_types:
                new_point = Point()
                new_point.coordinates = np.array([float(point['x']), float(point['y']), float(point['z'])])
                
                # remove pre-applied offsets from CIM valve points
                new_point.coordinates = new_point.coordinates - offsets[f"series_{point['series']}"][int(point['slice'])]
                
                for uid,frame in contours_CIM.frame.items():
                    if uid in uid_slice_match.keys():
                        if point['series'] == '0':
                            if uid_slice_match[uid] == int(point['slice']) and int(frame.time_frame) == int(point['frame']):
                                new_point.sop_instance_uid = uid
                                continue
                        elif point['series'] == '1':
                            if uid_slice_match[uid] == int(point['slice'])+sax_num and int(frame.time_frame) == int(point['frame']):
                                new_point.sop_instance_uid = uid
                                continue

                if new_point.sop_instance_uid == None:
                   print(f'No match found for slice {int(point["slice"])+sax_num} at frame {point["frame"]}')
                new_point.time_frame = int(point['frame'])
                try:
                    cont_name = valve_types[cim_valve_types.index(point['name'])]
                except:
                    cont_name = valve_types[cim_warp_types.index(point['name'])]
                contours_CIM.add_point(cont_name, new_point)

        print('Valve points added successfully')
    
    cvi_cont = CVI42XML()
    cvi_cont.contour = contours_CIM

    new_gpfilename = 'GPFile_cim.txt'
    new_metadatafilename = 'SliceInfoFile_cim.txt'
    output_gpfile = os.path.join(folder,new_gpfilename)
    output_metadatafile = os.path.join(folder,new_metadatafilename)
    cvi_cont.export_contour_points(output_gpfile)
    cvi_cont.export_dicom_metadata(output_metadatafile)

    return output_gpfile, output_metadatafile

def clean_CIM(folder, initial_gpfile, initial_sliceinfo, **kwargs):
    print("Cleaning contours...", end="")
    all_frames = pd.read_csv(os.path.join(folder, gpfile), sep="\t")
    frames_to_fit = sorted(np.unique([i[6] for i in all_frames.values]))
    filename = os.path.join(folder, gpfile)
    filenameInfo = os.path.join(folder, sliceinfofile)

    data_set = []
    for num in frames_to_fit:
        data_set.append(
            (
                num,
                GPDataSet(
                    filename, filenameInfo, case, sampling=1, time_frame_number=num
                ),
            )
        )

    final_data_set = Clean_contours(folder, data_set, "GPFile_CIM_clean.txt")
    print("done")

def extract_offsets(name):

    '''
    Code reads in Image Position Offset and Image Orientation Offset (from CIM) stored in OffsetFile for each slice of each case.
    Stores in a dictionary with key = slice number and value = offset
    Input: filename
    Output: dictionary of offsets
     
    '''
    offsets = {}
   #  it using csv read
    if not os.path.exists(name):
        return
    lines = []
    with open(name, 'rt') as in_file:
        for line in in_file:
            lines.append(re.split('\s+',line))
    
    try:
        index_slice = np.where(['slice' in x for x in lines[0]])[0][0]+1
        index_pos_offset = np.where(['ImagePositionOffset' in x for x in lines[0]])[0][0]+1
        index_ori_offset = np.where(['ImageOrientationOffset' in x for x in lines[0]])[0][0]+1
    except:
        index_slice = 0
        index_pos_offset = 0
        index_ori_offset = 0

    for i in range(0,len(lines)):
        slice_id = int(lines[i][index_slice])
        pos_offset = np.array(lines[i][index_pos_offset:index_pos_offset+3]).astype(
            float)
        ori_offset = np.array(lines[i][index_ori_offset:index_ori_offset+6]).astype(
            float)
        offsets.update({slice_id:[pos_offset,ori_offset]})

    return offsets    


def Landmarks_Dict(data_set, out_file, case):
    """
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

    """

    dataframe_dict = {
        "patient": [case],
        "frames": [],
        "MITRAL_VALVE": [],
        "TRICUSPID_VALVE": [],
        "AORTA_VALVE": [],
        "APEX_POINT": [],
        "status": [],
    }

    for idx, item in enumerate(data_set):
        frame_num = item[0]  # frame number
        data_set = item[1]  # GPDataset instance at given frame
        # load 3D coordinates for each landmark type
        mitral_points = data_set.points_coordinates[
            data_set.contour_type == ContourType.MITRAL_VALVE
        ]
        aorta_points = data_set.points_coordinates[
            data_set.contour_type == ContourType.AORTA_VALVE
        ]
        tricuspid_points = data_set.points_coordinates[
            data_set.contour_type == ContourType.TRICUSPID_VALVE
        ]
        apex_point = data_set.points_coordinates[
            data_set.contour_type == ContourType.APEX_POINT
        ]

        # save 3D coordinates at each frame for each landmark
        # dataframe_dict['patient'].append('')
        dataframe_dict["frames"].append(frame_num)
        dataframe_dict["MITRAL_VALVE"].append(mitral_points)
        dataframe_dict["TRICUSPID_VALVE"].append(tricuspid_points)
        dataframe_dict["AORTA_VALVE"].append(aorta_points)
        dataframe_dict["APEX_POINT"].append(apex_point)

    # count how many landmarks of each type are stored in GPFiles
    col1 = [k for k in dataframe_dict[list(fieldnames)[0]]]  # patient
    col2 = [k for k in dataframe_dict[list(fieldnames)[1]]]  # frames
    col3 = [len(k) for k in dataframe_dict[list(fieldnames)[2]]]  # mitral
    col4 = [len(k) for k in dataframe_dict[list(fieldnames)[3]]]  # tricuspid
    col5 = [len(k) for k in dataframe_dict[list(fieldnames)[4]]]  # aorta
    col6 = [len(k) for k in dataframe_dict[list(fieldnames)[5]]]  # apex

    with open(out_file, "a") as f:
        df = pd.DataFrame(
            list(itertools.zip_longest(*[col1, col2, col3, col4, col5, col6]))
        )
        df.to_csv(f, index=False, line_terminator="\n", header=False)

    # set expected umber of landmarks for each landmark type
    exp_values_dict = {
        "MITRAL_VALVE": (6, col3),
        "TRICUSPID_VALVE": (2, col4),
        "AORTA_VALVE": (2, col5),
        "APEX_POINT": (1, col6),
    }

    status = ["", ""]
    for label, items in exp_values_dict.items():
        exp_num = items[0]  # ecpected number of landmarks
        col = items[1]  # number of landmarks at each frame

        # look for missing landmarks
        if any(t < exp_num for t in col):
            masked_col = np.ma.getmask(np.ma.masked_less(col, exp_num))
            data = np.where(masked_col == True)[0]
            # count how many consecutive frames have missing landmarks
            count_dups = [len(list(group)) for group in mit.consecutive_groups(data)]
            # if up to 2 frames have missing landmarks, interpolate the values

            if any(t > 2 for t in count_dups):
                # delete dataset if landmarks are missing in 3 or more consecutive frames
                dataframe_dict = {"patient": case, "status": "deleted"}
                with open(out_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow("deleted")
                return dataframe_dict

            elif any(t <= 2 for t in count_dups):
                index = [i for (i, item) in enumerate(col) if item < exp_num]

                for idx in index:
                    try:
                        if col[idx - 1] != 0 and col[idx + 1] != 0:
                            avg_val = [
                                (g + h) / 2
                                for g, h in zip(
                                    dataframe_dict[label][idx - 1],
                                    dataframe_dict[label][idx + 1],
                                )
                            ]
                            dataframe_dict[label][idx] = np.array(avg_val)

                        elif col[idx - 1] != 0:
                            dataframe_dict[label][idx] = dataframe_dict[label][idx - 1]

                        elif col[idx + 1] != 0:
                            dataframe_dict[label][idx] = dataframe_dict[label][idx + 1]
                    except:
                        print("ERROR in dictionary creation")
                        ValueError

                    dataframe_dict["status"].append("changed")
                status.append("changed")

        else:
            dataframe_dict["status"].append("unchanged")
            status.append("unchanged")

            continue

    with open(out_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow(status)

    return dataframe_dict


def tracking_landmarks(folder, gpfilename, SliceInfoname, **kwargs):
    """
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

    """
    try:
        if "output_csv" in kwargs:
            landmarks_csv = kwargs.get("output_csv", None)

        final_status = "unchanged"
        filename = os.path.join(folder, gpfilename)
        filenameInfo = os.path.join(folder, SliceInfoname)

        # chose which frames to upload from the GPFile
        all_frames = pd.read_csv(filename, sep="\t")
        frames_to_fit = sorted(np.unique([i[6] for i in all_frames.values]))
        case = os.path.basename(os.path.normpath(folder))

        # build a list containing GPData structures, one for each frame
        print("-----> Uploading GP dataset")
        data_set = []  # structure (frame, GPData)
        for num in frames_to_fit:
            data_set.append(
                (
                    num,
                    GPDataSet(
                        filename, filenameInfo, case, sampling=1, time_frame_number=num
                    ),
                )
            )

        # Find missing landmarks and interpolate if necessary
        print("-----> Creating Dataframe")
        dataframe_dict = Landmarks_Dict(data_set, landmarks_csv, case)

        # find which frames have been changed by Landmarks_Dict()
        try:
            status = dataframe_dict["status"]
            contour_idx = [i for i, k in enumerate(status) if k == "changed"]
            if len(contour_idx) > 0:
                final_status = "changed"
        except:
            pass

        try:
            status = dataframe_dict["status"]
            if status == "deleted":
                final_status = "deleted"
        except:
            pass

        # write new GPFile
        dict_contypes = {
            "MITRAL_VALVE": ContourType.MITRAL_VALVE,
            "TRICUSPID_VALVE": ContourType.TRICUSPID_VALVE,
            "AORTA_VALVE": ContourType.AORTA_VALVE,
            "APEX_POINT": ContourType.APEX_POINT,
        }

        tracked_dataset = []
        for i, data in enumerate(data_set):  # this is iterating over time frames
            gpdata = data[1]
            if final_status == "deleted":
                continue
            elif final_status == "changed":
                for contour in contour_idx:
                    cont_name = fieldnames[contour + 2]
                    newpoints = dataframe_dict[cont_name][i]
                    # assign the new points to the datafile
                    try:
                        gpdata.points_coordinates[
                            gpdata.contour_type == dict_contypes[cont_name]
                        ] = newpoints
                    except:
                        slice_number = np.unique(
                            data_set[0][1].slice_number[
                                data_set[0][1].contour_type == dict_contypes[cont_name]
                            ]
                        )
                        for point in newpoints:
                            gpdata.add_data_points(
                                np.expand_dims(point, axis=0),
                                [dict_contypes[cont_name]],
                                slice_number,
                                [1],
                            )
                    tracked_dataset.append((data[0], gpdata))
            else:
                tracked_dataset.append(data)
                # clean points between the tricuspid and mitral valves
                # clean 3ch view (delete it because some 3ch points should be labelled as septum but they are not)

        return tracked_dataset

    except:
        return "ERROR"


def Clean_contours(folder, tracked_dataset, gpfilename_out, **kwargs):
    """
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

    """
    with open(os.path.join(folder, gpfilename_out), "w") as f:
        f.write(
            "{0:}\t{1}\t{2}\t".format("x", "y", "z")
            + "{0}\t".format("contour type")
            + "{0}\t{1}\t{2}".format("sliceID", "weight", "time frame")
            + "\n"
        )

    clean_dataset = []
    for i, data in enumerate(tracked_dataset):
        gpdata = data[1]
        gpdata.clean_LAX_contour()
        gpdata.write_gpfile(os.path.join(folder, gpfilename_out), time_frame=data[0])
        clean_dataset.append((data[0], gpdata))

    return clean_dataset


def findED_ESframe(case, data_set, **kwargs):
    """
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

    """

    if "case_frame_dict" in kwargs:
        case_frame_dict = kwargs.get("cases_frame_dict", None)
    else:
        case_frame_dict = {"case": []}
    
    if "dir_write" in kwargs:
        dir_write = kwargs.get("dir_write", None)

    dist_aorta_apex = []
    for data in data_set:
        gpdata = data[1]
        dist_aorta_apex.append(gpdata.dist_mitral_to_apex())

    if None in dist_aorta_apex:
        ED_frame = None
        ES_frame = None

    elif len(dist_aorta_apex) > 0:
        mindist_idx = np.argmin(dist_aorta_apex)
        ED_frame = 1
        ES_frame = data_set[mindist_idx][0]

    with open(dir_write, "a") as f:
        writer = csv.writer(f)
        writer.writerow((case, ED_frame, ES_frame))
