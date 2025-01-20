import os
import pydicom as dcm
import numpy as np

class Slice():
    def __init__(self,serie, slice, frame ,position, orientation, pixel_spacing,
                 image = None, subpixel_resolution = 1):
        self.position = position
        self.orientation = orientation
        self.pixel_spacing = pixel_spacing
        self.subpixel_resolution = subpixel_resolution
        self.serie = serie
        self.image = image

        self.time_frame = frame
        self.slice = slice


def extract_dicom_metadata(dicom_dir, dicom_extension='.dcm',  export_file =None):

    # search for dicom files asssociated with contours points
    file_uids = {}
    file_uid_data = {}

    for dir_name, subdir_list, file_list in os.walk(dicom_dir):
        total_files_uid = []
        instance_number = []
        position = []
        orientation = []
        pixel_spasing =[]
        series_number = []
        read = 0
        for filename in file_list:
            if dicom_extension in filename:  # check whether the file's DICOM
                file_dcm = os.path.join(dir_name, filename)
                dicom_data = dcm.read_file(file_dcm)
                file_uid = dicom_data.SOPInstanceUID

                position.append(np.asfarray(
                        dicom_data.ImagePositionPatient))
                orientation.append(np.asfarray(
                        dicom_data.ImageOrientationPatient
                    ))
                pixel_spasing.append(np.asfarray(
                        dicom_data.PixelSpacing
                    ))

                instance_number.append(int(dicom_data.InstanceNumber))
                series_number.append(float(dicom_data.SeriesNumber))
                read = read + 1
                total_files_uid.append(file_uid)

        #position = np.array(position)
        #sorted_position = np.unique(position, axis =0)
        #sorted_position = sorted_position[np.argsort(sorted_position[:,2])]

        unique_snb = np.unique(series_number)
        series_number = np.array(series_number)
        # in case that different series with different lengths are storen in
        # the same folder, the images stack needs to be sorted by series first

            # classify by series number
        for serie in unique_snb:
            serie_index = np.where(series_number == serie)[0]
            # the frames need to be computed using the full stack of SAx or LAX
            # images, for each folder the frame need to be computed
            # using the trigger time and instance number we recover the
            # slice images in the aquisition order
            serie_uids = np.array(
                [total_files_uid[x] for x in serie_index])
            serie_instance_nb = np.array(
                [instance_number[x] for x in serie_index])
            serie_position = np.array(
                [position[x] for x in serie_index])
            serie_orientation = np.array(
                [orientation[x] for x in serie_index]
            )
            serie_pixel = np.array(
                [pixel_spasing[x] for x in serie_index]
            )
            slice_len = int(np.sum(
                np.sum(np.equal(serie_position, serie_position[0]),
                       axis=1) == 3))
            sorted_position = np.unique(serie_position,axis = 0)

            for index, uid in enumerate(serie_uids):
                if not (uid in file_uid_data.keys()):
                    frame = int(serie_instance_nb[index] % slice_len)

                    if len(sorted_position) == 1:
                        slice = 0
                    else:
                        sorted_position = sorted_position[
                            np.argsort(sorted_position[:, 2])]
                        slice =np.where( np.sum(np.equal(sorted_position, serie_position[index]), axis = 1) == 3)[0][0]

                    if frame == 0:
                        frame = slice_len
                    image = Slice(serie, slice, frame ,
                                  serie_position[index],
                                  serie_orientation[index],
                                  serie_pixel[index])

                    file_uid_data.update({uid: image})
    
    if not (export_file is None):
        write_dicom_metadata(export_file, file_uid_data)
    return  file_uid_data


def write_dicom_metadata(file_name, dicom_metadata):
    #if not os.path.exists(os.path.dirname(file_name)):
        #os.makedirs(os.path.dirname(file_name))

    out = open(file_name, 'w')

    for uid in dicom_metadata.keys():
        position = dicom_metadata[uid].position
        orientation = dicom_metadata[uid].orientation
        spacing = dicom_metadata[uid].pixel_spacing
        frame_nb = dicom_metadata[uid].time_frame
        slice = dicom_metadata[uid].slice
        serie =dicom_metadata[uid].serie

        out.write(uid + '\tserie\t{0}\tslice\t{1}\ttimeFrame\t{2}'.format(serie,slice,frame_nb) +
                  '\tImagePositionPatient\t{0:.3f}\t{1:.3f}\t{'
                  '2:.3f}\t'.format(
                      position[0], position[1], position[2]) +
                  'ImageOrientationPatient\t{0:.3f}\t{1:.3f}\t{' \
                  '2:.3f}\t{3:.3f}\t{4:.3f}\t'
                  '{5:.3f}\t'.format(orientation[0], orientation[1],
                                     orientation[2], orientation[3],
                                     orientation[4],
                                     orientation[
                                         5]) + 'PixelSpacing\t{0:.3f}\t{' \
                                    '1:.3f}'.format(spacing[0], spacing[1]) + '\n')

if __name__ == '__main__':

    case = 'venus1'
    dicom_dir = 'C:/Users/ldt18/Desktop/empty1/'+case+'/dicoms'

    folderslist = [os.path.join(dicom_dir, folders) for folders in os.listdir(dicom_dir)]
    for folder in folderslist:
        extract_dicom_metadata(folder, 'dcm', 'C:/Users/ldt18/Desktop/empty/dicom_metadata.txt')
 
