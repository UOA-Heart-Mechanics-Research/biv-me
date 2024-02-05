from pathlib import Path
import pandas as pd
import numpy as np
import pydicom


if __name__ == "__main__":
    
    # set list of cases to process
    caselist = ["RV01", "RV02", "RV03", "RV04"]
    
    # set directory to save GP files
    dir_gp = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Debbie\collaborations\chicago-rv-mesh\analysis\gpfiles-raw"

    # set directory of corresponding DICOM images
    dir_dcm = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Debbie\collaborations\chicago-rv-mesh\images\chicago-cmr"
    dcm_extension = ".dcm"
    
    # -------------------------------------------------------------------
    
    for case in caselist:
        
        print(f"# Processing case {case}...")
        
        sliceinfo_path = Path(dir_gp, case, "SliceInfoFile.txt").as_posix()

        # get list of DICOM UIDs
        frames_uid = np.genfromtxt(sliceinfo_path, usecols=(0), dtype=str)
        
        # dataframe to store slice info
        df = pd.DataFrame(columns=["series_number", "frame", "uid", "filename"])
        
        # get list of DICOM files
        dcm_files = list(Path(dir_dcm, case).glob(f"**/*{dcm_extension}"))
        
        # loop through files and append to dataframe if UID matches
        num_files = len(dcm_files)
        
        for file in dcm_files:
            
            print(f"Scanning DICOM file ({dcm_files.index(file)+1}/{num_files})", end="\r")
            
            dcm = pydicom.read_file(file)
            if dcm.SOPInstanceUID in frames_uid:
                df.loc[len(df)] = [dcm.SeriesNumber, dcm.InstanceNumber, dcm.SOPInstanceUID, file.name]
                
        # sort dataframe by series number and frame
        df = df.sort_values(by=["series_number", "frame"])
        print(df)
        
        # get unique series numbers and number of frames per series
        series_numbers = df["series_number"].unique()
        num_frames = df.groupby("series_number").size().tolist()
        
        # print series numbers and corresponding number of frames
        print(f'Unique series:')
        [print(f'  Series {series_numbers[i]}: {num_frames[i]} frames') for i in range(len(series_numbers))]
        
        # save dataframe to csv
        # df.to_csv(Path(dir_gp, case, "dcm-series-info.csv"), index=False)
        # print(f"----- Saved {Path(dir_gp, case, 'dcm-series-info.csv')}\n")