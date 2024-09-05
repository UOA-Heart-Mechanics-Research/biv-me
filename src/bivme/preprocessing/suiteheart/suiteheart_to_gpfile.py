import os
import numpy as np
import glob
from pathlib import Path
import scipy.io as sio
import argparse
from loguru import logger


class Slice():
    def __init__(self, image_id,position, orientation, pixel_spacing,
                 sopUID=None,image = None, subpixel_resolution = 1):
        self.position = position
        self.orientation = orientation
        self.pixel_spacing = pixel_spacing
        self.subpixel_resolution = subpixel_resolution
        self.image = image

        self.time_frame = 1
        self.slice = None
        self.image_id = image_id
        self.sopUID = sopUID


    def get_affine_matrix(self, scaling = True):
        spacing = self.pixel_spacing
        image_position_patient = self.position
        image_orientation_patient = self.orientation
        # Translation
        T = np.identity(4)
        T[0:3, 3] = image_position_patient
        # Rotation
        R = np.identity(4)
        R[0:3, 0] = image_orientation_patient[0:3]
        R[0:3, 1] = image_orientation_patient[3:7]
        R[0:3, 2] = np.cross(R[0:3, 0], R[0:3, 1])
        T = np.dot(T, R)
        # scale
        if scaling:
            S = np.identity(4)
            S[0, 0] = spacing[1]
            S[1, 1] = spacing[0]
            T = np.dot(T, S)
        return T
    
def get_intersections(point_list1, point_list2, distance_cutoff=1):

    """ Finds the points that are within a given cutoff distance between two lists """

    a = range(len(point_list1))
    b = range(len(point_list2))
    [A, B] = np.meshgrid(a,b)
    c = np.concatenate([A.T, B.T], axis=0)
    pairs = c.reshape(2,-1).T
    dist = np.sqrt( ( ( point_list1[pairs[:,0],0] - point_list2[pairs[:,1],0] ) ** 2 ) + 
                    ( ( point_list1[pairs[:,0],1] - point_list2[pairs[:,1],1] ) ** 2 ) )
    
    pairs = pairs[np.where(dist < distance_cutoff)[0].tolist()]

    return pairs

def get_landmarks_from_intersections(point_list1, point_list2, distance_cutoff=1):
    while True:
        pairs = get_intersections(point_list1, point_list2, distance_cutoff)

        # If not enough intersections, increase distance cutoff
        if len(pairs) <= 2:
            distance_cutoff += 0.5
            continue
        
        # If not enough unique pairs, increase distance cutoff
        landmarkplane = point_list2[pairs[:,1],:]
        landmarkplane=np.unique(landmarkplane, axis=0)
        if len(landmarkplane) < 2:
            distance_cutoff += 0.5
            continue
        
        # Otherwise proceed
        break
    
    # Valve points will be extents of the intersection points
    landmarkplane = point_list2[pairs[:,1],:]
    landmarkplane=np.unique(landmarkplane, axis=0)
    # Find extent
    x = [v[0] for v in landmarkplane]
    y = [v[1] for v in landmarkplane]
    center = [np.mean(x), np.mean(y)]
    distances = [np.sqrt((v[0] - center[0])**2 + (v[1] - center[1])**2) for v in landmarkplane]
    landmarkplane = landmarkplane[np.argsort(distances)]
    radius = int(np.max(distances))

    landmarks = []
    
    max_dist = radius
    for i in range(len(landmarkplane)-len(landmarkplane//2), len(landmarkplane)):
        for j in range(i+1, len(landmarkplane)):
            dist = np.sqrt((landmarkplane[i][0] - landmarkplane[j][0])**2 + (landmarkplane[i][1] - landmarkplane[j][1])**2)
            if dist > radius:
                if dist > max_dist:
                    max_dist = dist
                    landmarks = np.array([landmarkplane[i], landmarkplane[j]], dtype=float)

    if len(landmarks)==0:
        pass
    return landmarks

def process_sax(saxfile):
    # Load sax matlab export
    sax_mat = sio.loadmat(saxfile)

    sax_contours = {'lv_endo': 'lv_endo', 'lv_epi': 'lv_epi', 'rv_endo': 'rv_endo'}

    num_phases = sax_mat['phase_number'][0][0]
    num_slices = sax_mat['slice_number'][0][0]

    sax_gp = []

    for phase in range(num_phases):
        # print(f"Processing frame {phase+1}")
        for slice in range(num_slices):
            # print(f"Processing slice {slice+1}")
            # Get img2world transform
            position = sax_mat['image_position'][phase][slice][0]
            orientation = sax_mat['orientation'][phase][slice][0]
            spacing = sax_mat['pixel_size'][phase][slice][0]
            slice_obj = Slice(slice, position, orientation, spacing)
            img2world = slice_obj.get_affine_matrix()

            # Check if contours exist
            if sax_mat[sax_contours['lv_endo']][phase][slice][0].shape[0] == 0:
                # print(f"Contour {sax_contours['lv_endo']} does not exist in frame {phase+1}, slice {slice+1}")
                # print(f"Skipping...")
                continue
            elif sax_mat[sax_contours['lv_epi']][phase][slice][0].shape[0] == 0:
                # print(f"Contour {sax_contours['lv_epi']} does not exist in frame {phase+1}, slice {slice+1}")
                # print(f"Skipping...")
                continue
            elif sax_mat[sax_contours['rv_endo']][phase][slice][0].shape[0] == 0:
                # print(f"Contour {sax_contours['rv_endo']} does not exist in frame {phase+1}, slice {slice+1}")
                # print(f"Skipping...")
                continue

            # Get contours
            lv_endo = sax_mat[sax_contours['lv_endo']][phase][slice][0]
            lv_endo = np.array([x.tolist() for i,x in enumerate(lv_endo[0]) if i % 2 == 0 ], dtype=float)[0]
            lv_epi = sax_mat[sax_contours['lv_epi']][phase][slice][0]
            lv_epi = np.array([x.tolist() for i,x in enumerate(lv_epi[0]) if i % 2 == 0 ], dtype=float)[0]
            rv_endo = sax_mat[sax_contours['rv_endo']][phase][slice][0]
            rv_endo = np.array([x.tolist() for i,x in enumerate(rv_endo[0]) if i % 2 == 0 ], dtype=float)[0]

            # Before transforming, apply pre-processing steps
            # Determine RV inserts
            rv_inserts = get_landmarks_from_intersections(rv_endo, lv_epi, distance_cutoff=1)

            # Split free wall and septum
            # Find pairs of points between rv endo and lv epi that are close to another
            cutoff = 1
            while True:
                pairs = get_intersections(rv_endo, lv_epi, distance_cutoff=cutoff)
                if len(pairs) > 0:
                    rv_septum = rv_endo[np.unique(pairs[:,0])] # deletes intersection from RV endo pts
                    rv_fw = np.array([pnt.tolist() for i, pnt in enumerate(rv_endo) if i not in np.unique(pairs[:,0])], 
                                            dtype=float)
                    lv_epi = np.array([pnt.tolist() for i, pnt in enumerate(lv_epi) if i not in np.unique(pairs[:,1])], 
                                            dtype=float)
                    break
                elif cutoff < 5:
                    cutoff += 0.5
                    continue
                else:
                    rv_septum = []
                    break
            


            points2D = [lv_endo, lv_epi, rv_fw, rv_septum, rv_inserts]
            points_contypes = ['SAX_LV_ENDOCARDIAL', 'SAX_LV_EPICARDIAL', 'SAX_RV_FREEWALL', 'SAX_RV_SEPTUM', 'RV_INSERT']
            points3D = []
            # Transform contours
            for i, points in enumerate(points2D):
                if len(points) > 0:
                    points = np.hstack((points, np.zeros((points.shape[0],1))))
                    points = np.hstack((points, np.ones((points.shape[0],1))))
                    points = np.dot(points, img2world.T)
                    points = points[:,:3]
                    points3D.append(points)
                else:
                    points3D.append([])

            # Save to sax gp
            for i, points in enumerate(points3D):
                if len(points) == 0:
                    continue
                for point in points:
                    string = f"{point[0]:.3f}\t{point[1]:.3f}\t{point[2]:.3f}\t{points_contypes[i]}\t{slice+1}\t1.0\t{phase+1}\n"
                    sax_gp.append(string)

    return sax_gp

def process_lax(laxfile, saxfile):
    # Load lax matlab export
    lax_mat = sio.loadmat(laxfile)

    lax_contours = {'lv_endo': 'lv_endo', 'lv_epi': 'lv_epi', 'rv_endo': 'rv_endo', 'la_endo': 'la_endo', 'ra_endo': 'ra_endo'}

    num_phases = lax_mat['phase_number'][0][0]
    num_slices = lax_mat['slice_number'][0][0]

    # Load sax matlab export
    sax_mat = sio.loadmat(saxfile)
    num_sax_slices = sax_mat['slice_number'][0][0]

    lax_gp = []

    for phase in range(num_phases):
        for slice in range(num_slices):
            # Get img2world transform
            position = lax_mat['image_position'][phase][slice][0]
            orientation = lax_mat['orientation'][phase][slice][0]
            spacing = lax_mat['pixel_size'][phase][slice][0]
            slice_obj = Slice(slice, position, orientation, spacing)
            img2world = slice_obj.get_affine_matrix()

            # Check if contours exist
            if lax_mat[lax_contours['lv_endo']][phase][slice][0].shape[0] == 0:
                # print(f"Contour {lax_contours['lv_endo']} does not exist in frame {phase+1}, slice {slice+1}")
                # print(f"Skipping...")
                continue
            elif lax_mat[lax_contours['lv_epi']][phase][slice][0].shape[0] == 0:
                # print(f"Contour {lax_contours['lv_epi']} does not exist in frame {phase+1}, slice {slice+1}")
                # print(f"Skipping...")
                continue

            # Get contours
            lv_endo = lax_mat[lax_contours['lv_endo']][phase][slice][0]
            lv_endo = np.array([x.tolist() for i,x in enumerate(lv_endo[0]) if i % 2 == 0 ], dtype=float)[0]
            lv_epi = lax_mat[lax_contours['lv_epi']][phase][slice][0]
            lv_epi = np.array([x.tolist() for i,x in enumerate(lv_epi[0]) if i % 2 == 0 ], dtype=float)[0]
            try:
                rv_endo = lax_mat[lax_contours['rv_endo']][phase][slice][0]
                rv_endo = np.array([x.tolist() for i,x in enumerate(rv_endo[0]) if i % 2 == 0 ], dtype=float)[0]
            except:
                rv_endo = []
            try:
                la_endo = lax_mat[lax_contours['la_endo']][phase][slice][0]
                la_endo = np.array([x.tolist() for i,x in enumerate(la_endo[0]) if i % 2 == 0 ], dtype=float)[0]
            except:
                la_endo = []
            try:
                ra_endo = lax_mat[lax_contours['ra_endo']][phase][slice][0]
                ra_endo = np.array([x.tolist() for i,x in enumerate(ra_endo[0]) if i % 2 == 0 ], dtype=float)[0]
            except:
                ra_endo = []
            
            # Before transforming, apply pre-processing steps
            # Split free wall and septum
            # Find pairs of points between rv endo and lv epi that are close to another
            rv_fw = []
            rv_septum = []
            cutoff = 1
            if len(rv_endo) > 0:
                while True:
                    pairs = get_intersections(rv_endo, lv_epi, distance_cutoff=cutoff)
                    if len(pairs) > 0:
                        rv_septum = rv_endo[np.unique(pairs[:,0])]
                        rv_fw = np.array([pnt.tolist() for i, pnt in enumerate(rv_endo) if i not in np.unique(pairs[:,0])], 
                                                dtype=float)
                        lv_epi = np.array([pnt.tolist() for i, pnt in enumerate(lv_epi) if i not in np.unique(pairs[:,1])],
                                                dtype=float)
                        break
                    elif cutoff < 5:
                        cutoff += 0.5
                        continue
                    else:
                        rv_septum = []
                        break
                
            # Determine mitral valve points
            # Present on all slices (2ch, 3ch, 4ch)
            # Attempt to extract from MV annulus in LAX matlab export
            try:
                mv1 = lax_mat['mv_annulus'][phase][slice][0][0][2][0]
                mv2 = lax_mat['mv_annulus'][phase][slice][0][0][2][1]
                mv=np.array([mv1,mv2])
            except:
                logger.info(f"MV annulus not found in {laxfile}")
                # Try to extract from intersection of LV endo and LA endo
                if len(la_endo) > 0:
                    mv = get_landmarks_from_intersections(lv_endo, la_endo, distance_cutoff=1)
                else:
                    mv = []
            
            if len(rv_endo) > 0: # therefore it is a 4Ch slice -> get lv epi apex and tv points
                # Determine LV apex
                mv_centroid = np.mean(mv, axis=0)
                # Get lv epi point furthest from centroid
                distances = [np.sqrt((v[0] - mv_centroid[0])**2 + (v[1] - mv_centroid[1])**2) for v in lv_epi]
                lv_apex = lv_epi[np.argmax(distances)]

                # Determine tricuspid valve points
                # Attempt to extract from TV annulus in LAX matlab export
                try:
                    tv1 = lax_mat['tv_annulus'][phase][slice][0][0][2][0]
                    tv2 = lax_mat['tv_annulus'][phase][slice][0][0][2][1]
                    tv=np.array([tv1,tv2])
                except:
                    logger.info(f"TV annulus not found in {laxfile}")
                    # Try to extract from intersection of RV endo and RA endo
                    if len(ra_endo) > 0:
                        tv = get_landmarks_from_intersections(rv_endo, ra_endo, distance_cutoff=1)
                    else:
                        tv = []
            else:
                lv_apex = []
                tv = []
            
            points2D = [lv_endo, lv_epi, rv_fw, rv_septum, mv, tv, lv_apex]
            points_contypes = ['LAX_LV_ENDOCARDIAL', 'LAX_LV_EPICARDIAL', 'LAX_RV_FREEWALL', 'LAX_RV_SEPTUM', 'MITRAL_VALVE', 'TRICUSPID_VALVE', 'APEX_POINT']
            points3D = []
            # Transform contours
            for i, points in enumerate(points2D):
                if len(points) > 0:
                    if points.size == 2:
                        points = np.array([points[0], points[1], 0, 1])
                        points = np.dot(points, img2world.T)
                        points = points[:3]
                    else:
                        points = np.hstack((points, np.zeros((points.shape[0],1))))
                        points = np.hstack((points, np.ones((points.shape[0],1))))
                        points = np.dot(points, img2world.T)
                        points = points[:,:3]
                    points3D.append(points)
                else:
                    points3D.append([])

            # Save to lax gp
            for i, points in enumerate(points3D):
                if len(points) == 0:
                    continue
                elif points.size == 3:
                    string = f"{points[0]:.3f}\t{points[1]:.3f}\t{points[2]:.3f}\t{points_contypes[i]}\t{slice+num_sax_slices+1}\t1.0\t{phase+1}\n"
                    lax_gp.append(string)
                else:
                    for point in points:
                        string = f"{point[0]:.3f}\t{point[1]:.3f}\t{point[2]:.3f}\t{points_contypes[i]}\t{slice+num_sax_slices+1}\t1.0\t{phase+1}\n"
                        lax_gp.append(string)

    return lax_gp

def write_slice_info_file(saxfile,laxfile, sliceinfofile):
    # Find all slices
    sax_mat = sio.loadmat(saxfile)
    lax_mat = sio.loadmat(laxfile)

    num_sax_slices = sax_mat['slice_number'][0][0]
    num_lax_slices = lax_mat['slice_number'][0][0]

    for slice in range(num_sax_slices+num_lax_slices):
        if slice < num_sax_slices:
            position = sax_mat['image_position'][0][slice][0]
            orientation = sax_mat['orientation'][0][slice][0]
            spacing = sax_mat['pixel_size'][0][slice][0]
            uid = sax_mat['uid'][0][slice][0]
        else:
            position = lax_mat['image_position'][0][slice-num_sax_slices][0]
            orientation = lax_mat['orientation'][0][slice-num_sax_slices][0]
            spacing = lax_mat['pixel_size'][0][slice-num_sax_slices][0]
            uid = lax_mat['uid'][0][slice-num_sax_slices][0]
        
        string = f"{uid}\tframeID:\t{slice+1}\tImagePositionPatient\t{position[0]:.3f}\t{position[1]:.3f}\t{position[2]:.3f}\tImageOrientationPatient\t{orientation[0]:.3f}\t{orientation[1]:3f}\t{orientation[2]:.3f}\t{orientation[3]:.3f}\t{orientation[4]:.3f}\t{orientation[5]:.3f}\tPixelSpacing\t{spacing[0]:.3f}\t{spacing[1]:.3f}\n"
        
        # Append to slice info file
        with open(sliceinfofile, 'a') as f:
            f.write(string)
    
    

if __name__ == "__main__":
    '''
    Converts between SuiteHeart .mat files and GPFile.txt and SliceInfoFile.txt for use in biv fitting

    Author: Joshua Dillon
    Last updated: 2024-08-08
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', action='store', help='path to directory containing .mat files exported from suiteheart')
    parser.add_argument('--out-dir', '-o', action='store', help='path to output directory for gpfile and sliceinfofile')
    args = parser.parse_args()

    dir_mat = args.data_dir
    dir_out = args.out_dir

    caselist = os.listdir(dir_mat)
    casedirs = [Path(dir_mat, case).as_posix() for case in caselist]

    for folder in casedirs:
        logger.info(f"Processing {os.path.basename(folder)}")

        if not os.path.exists(os.path.join(dir_out, os.path.basename(folder))):
            os.makedirs(os.path.join(dir_out, os.path.basename(folder)))

        # Find SAX and LAX files
        matfiles = glob.glob(os.path.join(folder, "*.mat"))
        laxfile = [f for f in matfiles if "LAX" in f][0]
        saxfile = [f for f in matfiles if "LAX" not in f][0]

        sax_gp = process_sax(saxfile)
        lax_gp = process_lax(laxfile, saxfile)

        # Write GPFile
        gpfile = os.path.join(dir_out, os.path.basename(folder))
        assert os.path.exists(gpfile), f"Cannot write to {os.path.join(gpfile, 'GPFile.txt')}!"
        gpfile = os.path.join(gpfile, "GPFile.txt")

        with open(gpfile, 'w') as f:
            f.write("x\ty\tz\tcontour type\tsliceID\tweight\ttime frame\n")
            for line in sax_gp:
                f.write(line)
            for line in lax_gp:
                f.write(line)
        
        logger.success(f"Saved GPFile to {gpfile}")

        # Write slice info file
        sliceinfofile = os.path.join(dir_out, os.path.basename(folder))
        assert os.path.exists(sliceinfofile), f"Cannot write to {os.path.join(sliceinfofile, 'SliceInfoFile.txt')}!"
        sliceinfofile = os.path.join(sliceinfofile, "SliceInfoFile.txt")

        write_slice_info_file(saxfile,laxfile, sliceinfofile)
        
        logger.success(f"Saved SliceInfoFile to {sliceinfofile}")