import os
import numpy as np

def write_to_gp_file(path, coords, label, slice_id, weight=1.0, phase=1.0):

    """ Writes a coordinate/line to the guide point file """
    
    # check if output exists
    if os.path.exists(path):
        flag = 'a'
    else:
        flag = 'w'
    
    count = 0
    while count<100000000:
        try:
            with open(os.path.join(path), flag) as f:
                if flag == 'w':
                    f.write('x\ty\tz\tcontour type\tframeID\tweight\ttime frame\n')
                for coord in coords:
                    f.write('{:.5f}\t{:.5f}\t{:.5f}\t{}\t{}\t{}\t{}\n'.format(coord[0], coord[1], coord[2],
                                                                label, slice_id, weight, phase))
                break
        except:
            count += 1
            pass

def inverse_coordinate_transformation(coordinate, imagePositionPatient, imageOrientationPatient, ps):

    """ Performs a coordinate transformation from image coordinates to patient coordinates """

    # image position and orientation
    S = imagePositionPatient
    X = imageOrientationPatient[:3]
    Y = imageOrientationPatient[3:]

    # construct affine transform
    M = np.asarray([[X[0]*ps[0], Y[0]*ps[1], 0, S[0]],
                [X[1]*ps[0], Y[1]*ps[1], 0, S[1]],
                [X[2]*ps[0], Y[2]*ps[1], 0, S[2]],
                [0, 0, 0, 1]])

    coord = np.array([coordinate[1], coordinate[0], 0, 1.0])
    
    # perform transformation and return as list
    return [np.round(x,5) for x in M @ coord.T]


