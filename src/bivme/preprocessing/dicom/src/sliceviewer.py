import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

import bivme.preprocessing.dicom.src.contouring as contouring
import bivme.preprocessing.dicom.src.guidepointprocessing as guidepointprocessing

class SliceViewer:
    def __init__(self, patient, processed_folder, slice_info_df, view, sliceID, es_phase, num_phases, full_cycle=True, version='2d'):
        
        self.slice_info_df = slice_info_df
        self.view = view
        self.sliceID = sliceID
        self.es_phase = int(es_phase)
        self.num_phases = num_phases
        if full_cycle:
            self.phases = np.arange(0,self.num_phases, dtype=int)
        else:
            self.phases = [0, self.es_phase]
        self.slice = self.slice_info_df[(self.slice_info_df['View'] == self.view) & (self.slice_info_df['Slice ID'] == self.sliceID)]
        self.patient = patient
        self.image_folder = os.path.join(processed_folder, patient, 'images') # Only for plotting & debugging
        self.segmentation_folder = os.path.join(processed_folder, patient, 'segmentations')
        self.version = version  # 2D or 3D
        # self.images = self.get_images()
        self.segmentations = self.get_segmentations()
        self.get_contours()
        self.landmarks = None

    def get_images(self):
        images = []
        for phase in self.phases:
            image = os.path.join(self.image_folder, self.view, f"{self.view}_2d_{self.sliceID}_{int(phase):03}_0000.nii.gz")
            image = nib.load(image).get_fdata()
            image = np.transpose(image)
            images.append(image)

        self.size = images[0].shape
        return images
        
    def get_segmentations(self):
        segmentations = []
        for phase in self.phases:
            segmentation = os.path.join(self.segmentation_folder, self.view, f"{self.view}_2d_{self.sliceID}_{int(phase):03}.nii.gz")
            segmentation = nib.load(segmentation).get_fdata()
            segmentation = np.transpose(segmentation)
            segmentations.append(segmentation)
        
        self.size = segmentations[0].shape
        return segmentations
    
    def get_initial_landmarks(self):
        self.get_landmarks_from_intersections()

        self.landmarks = {
            'SAX': {'RVI': {}},
            '2ch': {'MV': {}},
            '3ch': {'MV': {}, 'AV': {}},
            '4ch': {'MV': {}, 'TV': {}, 'LVA': {}},
            'RVOT': {'PV': {}}
        }

        if self.view == 'SAX':
            for phase in self.phases:
                self.landmarks['SAX']['RVI'][f'{phase}'] = self.rvi[self.view][f'{phase}']

        elif self.view == '2ch':
            for phase in self.phases:
                self.landmarks['2ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']
        
        elif self.view == '3ch':
            for phase in self.phases:
                self.landmarks['3ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']
                self.landmarks['3ch']['AV'][f'{phase}'] = self.av[self.view][f'{phase}']

        elif self.view == '4ch':
            for phase in self.phases:
                self.landmarks['4ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']
                self.landmarks['4ch']['TV'][f'{phase}'] = self.tv[self.view][f'{phase}']
                self.landmarks['4ch']['LVA'][f'{phase}'] = self.lva[self.view][f'{phase}']

        elif self.view == 'RVOT':
            for phase in self.phases:
                self.landmarks['RVOT']['PV'][f'{phase}'] = self.pv[self.view][f'{phase}']

        return self.landmarks

    def update_landmarks(self, landmarks_df):
        self.get_landmarks_from_df(landmarks_df)

        self.landmarks = {}

        if self.view == 'SAX':
            for phase in self.phases:
                self.landmarks['SAX']['RVI'][f'{phase}'] = self.rvi[self.view][f'{phase}']

        elif self.view == 'RVOT':
            for phase in self.phases:
                self.landmarks['RVOT']['PV'][f'{phase}'] = self.pv[self.view][f'{phase}']

        elif self.view == '2ch':
            for phase in self.phases:
                self.landmarks['2ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']

        elif self.view == '3ch':
            for phase in self.phases:
                self.landmarks['3ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']
                self.landmarks['3ch']['AV'][f'{phase}'] = self.av[self.view][f'{phase}']

        elif self.view == '4ch':
            for phase in self.phases:
                self.landmarks['4ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']
                self.landmarks['4ch']['TV'][f'{phase}'] = self.tv[self.view][f'{phase}']
                self.landmarks['4ch']['LVA'][f'{phase}'] = self.lva[self.view][f'{phase}']

    def get_landmarks_from_intersections(self):

        self.rvi = {'SAX': {}}
        self.mv = {'2ch': {}, '3ch': {}, '4ch': {}}
        self.av = {'3ch': {}}
        self.tv = {'4ch': {}}
        self.lva = {'4ch': {}}
        self.pv = {'RVOT': {}}

        if self.view == 'SAX':
            for i, phase in enumerate(self.phases):
                try:
                    self.rvi['SAX'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 2, 3, distance_cutoff=3.5)
                except:
                    self.rvi['SAX'][f'{phase}'] = None

        elif self.view == '2ch':
            for i, phase in enumerate(self.phases):
                try:
                    self.mv['2ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 3)
                except:
                    self.mv['2ch'][f'{phase}'] = None
        
        elif self.view == '3ch':
            for i, phase in enumerate(self.phases):
                try:
                    self.mv['3ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 4)
                except:
                    self.mv['3ch'][f'{phase}'] = None
                
                try:
                    self.av['3ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 5)
                except:
                    self.av['3ch'][f'{phase}'] = None

        elif self.view == '4ch':
            for i, phase in enumerate(self.phases):
                try:
                    self.mv['4ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 4)
                except:
                    self.mv['4ch'][f'{phase}'] = None
                
                try:
                    self.tv['4ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 3, 5)
                except:
                    self.tv['4ch'][f'{phase}'] = None

                try:
                    self.lva['4ch'][f'{phase}'] = contouring.estimate_lva(self.contours[f'{phase}'][1], self.mv['4ch'][f'{phase}'][0],  self.mv['4ch'][f'{phase}'][1])
                except:
                    self.lva['4ch'][f'{phase}'] = None
                    print(f'LVA not found on slice {self.sliceID} phase {phase}')

        elif self.view == 'RVOT':
            for i, phase in enumerate(self.phases):
                try:
                    self.pv['RVOT'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 3)
                except:
                    self.pv['RVOT'][f'{phase}'] = None
    
        else:
            print('View not supported for valve detection')
    
    def get_landmarks_from_df(self, landmarks_df):
        self.rvi = {}
        self.mv = {}
        self.av = {}
        self.tv = {}
        self.lva = {}
        self.pv = {}

        if self.view == 'SAX':
            SAX_df = landmarks_df[landmarks_df['View'] == 'SAX']
            SAX_df = SAX_df[SAX_df['Slice ID'] == self.sliceID]
            for i, phase in enumerate(self.phases):
                try:
                    rvi = np.array([np.array(SAX_df['RV1'].values[i]), np.array(SAX_df['RV2'].values[i])])
                except:
                    rvi = None
                self.rvi[self.view][f'{phase}'] = rvi

        elif self.view == '4ch':
            four_chamber_df = landmarks_df[landmarks_df['View'] == '4ch']
            four_chamber_df = four_chamber_df[four_chamber_df['Slice ID'] == self.sliceID]
            for i, phase in enumerate(self.phases):
                try:
                    mv = np.array([np.array(four_chamber_df['MV1'].values[i]), np.array(four_chamber_df['MV2'].values[i])])
                except:
                    mv = None
                try:
                    tv = np.array([np.array(four_chamber_df['TV1'].values[i]), np.array(four_chamber_df['TV2'].values[i])])
                except:
                    tv = None
                try:
                    lva = np.array(four_chamber_df['LVA'].values[i])
                except:
                    lva = None

                self.mv[self.view][f'{phase}'] = mv
                self.tv[self.view][f'{phase}'] = tv
                self.lva[self.view][f'{phase}'] = lva
        
        elif self.view == '3ch':
            three_chamber_df = landmarks_df[landmarks_df['View'] == '3ch']
            three_chamber_df = three_chamber_df[three_chamber_df['Slice ID'] == self.sliceID]
            for i, phase in enumerate(self.phases):
                try:
                    mv = np.array([np.array(three_chamber_df['MV1'].values[i]), np.array(three_chamber_df['MV2'].values[i])])
                except:
                    mv = None
                try:
                    av = np.array([np.array(three_chamber_df['AV1'].values[i]), np.array(three_chamber_df['AV2'].values[i])])
                except:
                    av = None
                
                self.mv[self.view][f'{phase}'] = mv
                self.av[self.view][f'{phase}'] = av
        
        elif self.view == '2ch':
            two_chamber_df = landmarks_df[landmarks_df['View'] == '2ch']
            two_chamber_df = two_chamber_df[two_chamber_df['Slice ID'] == self.sliceID]
            for i, phase in enumerate(self.phases):
                try:
                    mv = np.array([np.array(two_chamber_df['MV1'].values[i]), np.array(two_chamber_df['MV2'].values[i])])
                except:
                    mv = None

                self.mv[self.view][f'{phase}'] = mv
        
        elif self.view == 'RVOT':
            rvot_df = landmarks_df[landmarks_df['View'] == 'RVOT']
            rvot_df = rvot_df[rvot_df['Slice ID'] == self.sliceID]
            for i, phase in enumerate(self.phases):
                try:
                    pv = np.array([np.array(rvot_df['PV1'].values[i]), np.array(rvot_df['PV2'].values[i])])
                except:
                    pv = None

                self.pv[self.view][f'{phase}'] = pv
        
    def get_contours(self):
        ## TODO: QC 
        contours = {}

        if self.view == 'SAX':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_SAX(self.segmentations[i])
        elif self.view == 'RVOT':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_RVOT(self.segmentations[i])
        elif self.view == '2ch':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_2ch(self.segmentations[i])
        elif self.view == '3ch':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_3ch(self.segmentations[i])
        elif self.view == '4ch':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_4ch(self.segmentations[i])
        
        self.contours = contours

    # TODO: Remove this - redundant
    def clean_lax(self):
        if self.view == '4ch':
            # remove points that are not on the endocardial contours
            for phase in self.phases:
                LV_endo = self.contours[str(phase)][0]
                LV_epi = self.contours[str(phase)][1]
                RV_endo = self.contours[str(phase)][0]
                RV_septal = self.contours[str(phase)][2]
                RV_epi = self.contours[str(phase)][1]


                MV_pts = self.mv[self.view][str(phase)]
                TV_pts = self.tv[self.view][str(phase)]

                # Clean between mitral valve points
                del_idx = []
                for i in range(len(LV_endo)):
                    point = LV_endo[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_endo = np.delete(LV_endo, del_idx, axis=0)
                self.contours[str(phase)][4] = LV_endo

                del_idx = []
                for i in range(len(LV_epi)):
                    point = LV_epi[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_epi = np.delete(LV_epi, del_idx, axis=0)
                self.contours[str(phase)][3] = LV_epi

                # Clean between tricuspid valve points
                del_idx = []
                for i in range(len(RV_endo)):
                    point = RV_endo[i]
                    valve_dist = np.linalg.norm(TV_pts[1] - TV_pts[0])
                    distance1 = np.linalg.norm(TV_pts[1] - point)
                    distance2 = np.linalg.norm(TV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                RV_endo = np.delete(RV_endo, del_idx, axis=0)
                self.contours[str(phase)][0] = RV_endo

                del_idx = []
                for i in range(len(RV_septal)):
                    point = RV_septal[i]
                    valve_dist = np.linalg.norm(TV_pts[1] - TV_pts[0])
                    distance1 = np.linalg.norm(TV_pts[1] - point)
                    distance2 = np.linalg.norm(TV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                RV_septal = np.delete(RV_septal, del_idx, axis=0)
                self.contours[str(phase)][2] = RV_septal

        elif self.view == '3ch':
            # remove points that are not on the endocardial contours
            for phase in self.phases:
                RV_endo = self.contours[str(phase)][0]
                RV_epi = self.contours[str(phase)][1]
                RV_septal = self.contours[str(phase)][2]
                LV_epi = self.contours[str(phase)][3]
                LV_endo = self.contours[str(phase)][4]

                MV_pts = self.mv[self.view][str(phase)]
                AV_pts = self.av[self.view][str(phase)]

                # Clean between mitral valve points
                del_idx = []
                for i in range(len(LV_endo)):
                    point = LV_endo[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_endo = np.delete(LV_endo, del_idx, axis=0)
                self.contours[str(phase)][4] = LV_endo

                del_idx = []
                for i in range(len(LV_epi)):
                    point = LV_epi[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_epi = np.delete(LV_epi, del_idx, axis=0)
                self.contours[str(phase)][3] = LV_epi

                # Clean between aortic valve points
                del_idx = []
                for i in range(len(LV_endo)):
                    point = LV_endo[i]
                    valve_dist = np.linalg.norm(AV_pts[1] - AV_pts[0])
                    distance1 = np.linalg.norm(AV_pts[1] - point)
                    distance2 = np.linalg.norm(AV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_endo = np.delete(LV_endo, del_idx, axis=0)
                self.contours[str(phase)][4] = LV_endo

                del_idx = []
                for i in range(len(LV_epi)):
                    point = LV_epi[i]
                    valve_dist = np.linalg.norm(AV_pts[1] - AV_pts[0])
                    distance1 = np.linalg.norm(AV_pts[1] - point)
                    distance2 = np.linalg.norm(AV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)

                LV_epi = np.delete(LV_epi, del_idx, axis=0)
                self.contours[str(phase)][3] = LV_epi

        elif self.view == '2ch':
            # remove points that are not on the endocardial contours
            for phase in self.phases:
                LV_endo = self.contours[str(phase)][0]
                LV_epi = self.contours[str(phase)][1]

                MV_pts = self.mv[self.view][str(phase)]

                # Clean between mitral valve points
                del_idx = []
                for i in range(len(LV_endo)):
                    point = LV_endo[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_endo = np.delete(LV_endo, del_idx, axis=0)
                self.contours[str(phase)][0] = LV_endo

                del_idx = []
                for i in range(len(LV_epi)):
                    point = LV_epi[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_epi = np.delete(LV_epi, del_idx, axis=0)
                self.contours[str(phase)][1] = LV_epi
        
        elif self.view == 'RVOT':
            # remove points that are not on the endocardial contours
            for phase in self.phases:
                RV_endo = self.contours[str(phase)][0]
                RV_septal = self.contours[str(phase)][1]

                PV_pts = self.pv[self.view][str(phase)]

                # Clean between pulmonary valve points
                del_idx = []
                for i in range(len(RV_endo)):
                    point = RV_endo[i]
                    valve_dist = np.linalg.norm(PV_pts[1] - PV_pts[0])
                    distance1 = np.linalg.norm(PV_pts[1] - point)
                    distance2 = np.linalg.norm(PV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                RV_endo = np.delete(RV_endo, del_idx, axis=0)
                self.contours[str(phase)][0] = RV_endo

                del_idx = []
                for i in range(len(RV_septal)):
                    point = RV_septal[i]
                    valve_dist = np.linalg.norm(PV_pts[1] - PV_pts[0])
                    distance1 = np.linalg.norm(PV_pts[1] - point)
                    distance2 = np.linalg.norm(PV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                RV_septal = np.delete(RV_septal, del_idx, axis=0)
                self.contours[str(phase)][1] = RV_septal


    def get_slice_info(self):
        self.imgPos = self.slice['ImagePositionPatient'].values[0]
        self.imgOrient = self.slice['ImageOrientationPatient'].values[0]
        self.ps = self.slice['Pixel Spacing'].values[0]

    def export_slice(self, output_folder, slice_mapping):
        self.get_slice_info()
        os.makedirs(output_folder, exist_ok=True)

        self.mapped_sliceID = slice_mapping[self.sliceID]
        
        if self.view == 'SAX':
            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                RV_septal_pts = self.contours[phase][2]
                RV_fw_pts = self.contours[phase][3]
                RV_epi_pts = self.contours[phase][4]

                RVI_pts = self.rvi[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                                RV_epi_pts,
                                RV_septal_pts,
                                LV_epi_pts,
                                LV_endo_pts,
                                RVI_pts]
                
                labels = ['SAX_RV_FREEWALL',
                            'SAX_RV_EPICARDIAL',
                            'SAX_RV_SEPTUM',
                            'SAX_LV_EPICARDIAL',
                            'SAX_LV_ENDOCARDIAL',
                            'RV_INSERT']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue
                    
                    if len(points) == 0:
                        continue
                        
                    # Invert points first due to initial transpose
                    points = [np.flip(point) for point in points]
                    pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                            for point in points]

                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.mapped_sliceID, weight=1.0, phase=int(phase))

        elif self.view == 'RVOT':
            for phase in self.phases:
                phase = str(phase)

                RV_s_pts = self.contours[phase][0]
                RV_fw_pts = self.contours[phase][1]
                RV_epi_pts = self.contours[phase][2]
                pa_pts = self.contours[phase][3]

                PV_pts = self.pv[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                                RV_s_pts,
                                RV_epi_pts,
                                PV_pts]
                
                labels = ['SAX_RV_FREEWALL',
                            'SAX_RV_SEPTUM',
                            'SAX_RV_EPICARDIAL',
                            'PULMONARY_VALVE']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points) == 0:
                        continue

                    points = [np.flip(point) for point in points]
                    pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                            for point in points]
                            
                    # Write to file
                    
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.mapped_sliceID, weight=1.0, phase=int(phase))

        elif self.view == '2ch':
            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                la_pts = self.contours[phase][2]

                MV_pts = self.mv[self.view][str(phase)]

                point_lists = [LV_endo_pts, 
                                LV_epi_pts,
                                MV_pts]
                
                labels = ['LAX_LV_ENDOCARDIAL',
                            'LAX_LV_EPICARDIAL',
                            'MITRAL_VALVE']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points) == 0:
                        continue

                    points = [np.flip(point) for point in points]
                    pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                            for point in points]
                    
                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.mapped_sliceID, weight=1.0, phase=int(phase))

        elif self.view == '3ch':
            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                RV_septal_pts = self.contours[phase][2]
                RV_fw_pts = self.contours[phase][3]
                RV_epi_pts = self.contours[phase][6]

                MV_pts = self.mv[self.view][str(phase)]
                AV_pts = self.av[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                               RV_epi_pts,
                               RV_septal_pts,
                               LV_epi_pts,
                               LV_endo_pts,
                               RV_epi_pts,
                               MV_pts,
                               AV_pts]
                
                labels = ['LAX_RV_FREEWALL',
                            'LAX_RV_EPICARDIAL',
                            'LAX_RV_SEPTUM',
                            'LAX_LV_EPICARDIAL',
                            'LAX_LV_ENDOCARDIAL',
                            'LAX_RV_EPICARDIAL',
                            'MITRAL_VALVE',
                            'AORTA_VALVE']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points)== 0:
                        continue

                    points = [np.flip(point) for point in points]
                    pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                            for point in points]
                    
                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.mapped_sliceID, weight=1.0, phase=1.0)

        elif self.view == '4ch':
            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                RV_septal_pts = self.contours[phase][2]
                RV_fw_pts = self.contours[phase][3]
                RV_epi_pts = self.contours[phase][6]



                MV_pts = self.mv[self.view][str(phase)]
                TV_pts = self.tv[self.view][str(phase)]
                LVA_pts = self.lva[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                               RV_epi_pts,
                               RV_septal_pts,
                               LV_epi_pts,
                               LV_endo_pts,
                               RV_epi_pts,
                               MV_pts,
                               TV_pts,
                               LVA_pts]
                
                labels = ['LAX_RV_FREEWALL',
                            'LAX_RV_EPICARDIAL',
                            'LAX_RV_SEPTUM',
                            'LAX_LV_EPICARDIAL',
                            'LAX_LV_ENDOCARDIAL',
                            'LAX_RV_EPICARDIAL',
                            'MITRAL_VALVE',
                            'TRICUSPID_VALVE',
                            'APEX_POINT']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points) == 0:
                        continue

                    elif points.size == 2:
                        points = np.flip(points)
                        pts = [guidepointprocessing.inverse_coordinate_transformation(points, self.imgPos, self.imgOrient, self.ps)]

                    else:
                        points = [np.flip(point) for point in points]
                        pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                                for point in points]

                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.mapped_sliceID, weight=1.0, phase=int(phase))

    def plot_slice(self):
        import matplotlib
        matplotlib.use('TkAgg')
        plt.ion()

        fig, ax = plt.subplots(2, 2, figsize=(15, 8))
        # image and segmentation for each time frame
        ax[0, 0].imshow(self.images[0], cmap='gray')
        ax[0, 0].imshow(self.segmentations[0], cmap='inferno', alpha=0.2)
        ax[0, 0].set_title(f'{self.view} Slice {self.sliceID} Time Frame 0')
        ax[0, 0].axis('off')
        ax[0, 1].imshow(self.images[1], cmap='gray')
        ax[0, 1].imshow(self.segmentations[1], cmap='inferno', alpha=0.2)
        ax[0, 1].set_title(f'{self.view} Slice {self.sliceID} Time Frame {int(self.es_phase)}')
        ax[0, 1].axis('off')

        # landmarks and contours
        ax[1, 0].imshow(self.images[0], cmap='gray')
        ax[1, 1].imshow(self.images[1], cmap='gray')
        colours = ['r', 'g', 'b', 'y', 'm']

        for i, phase in enumerate(self.contours):
            for j, contype in enumerate(self.contours[phase]):
                try:
                    ax[1, i].scatter(contype[:,0], contype[:,1], c=colours[j], s=1)
                except:
                    pass
        
        try:
            for i, phase in enumerate([0, int(self.es_phase)]):
                for j, landmarktype in enumerate(self.landmarks[self.view]):
                    try:
                        ax[1, i].scatter(self.landmarks[self.view][landmarktype][str(phase)][0][0], self.landmarks[self.view][landmarktype][str(phase)][0][1], c='white', marker = '+', s=5)
                        ax[1, i].scatter(self.landmarks[self.view][landmarktype][str(phase)][1][0], self.landmarks[self.view][landmarktype][str(phase)][1][1], c='white', marker = '+', s=5)
                    except:
                        try:
                            ax[1, i].scatter(self.landmarks[self.view][landmarktype][str(phase)][0], self.landmarks[self.view][landmarktype][str(phase)][1], c='white', marker = '+', s=5)
                        except:
                            pass
                        pass
        except:
            pass
        plt.show(block=False)
        plt.pause(0.01)


        

        
    


    