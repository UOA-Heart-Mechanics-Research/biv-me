import numpy as np
import sys
import time
import functools
import functions
import pandas as pd
import importlib
import math
importlib.reload(functions)
from functions import*
import re
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
##Author : Charlène Mauger, University of Auckland, c.mauger@auckland.ac.nz
class GPDataSet(object):

    """ This class reads a dataset. A DataSet object has the following properties: 

    Attributes:
        case: case name
        Mitral_centroid: centroid of the 3D contour points labelled as mitral valve
        Tricuspid_centroid: centroid of the 3D contour points labelled as tricuspid valve
        Aorta_centroid: centroid of the 3D contour points labelled as aortic valve
        Pulmonary_centroid: centroid of the 3D contour points labelled as pulmonic valve
        number_of_slice: number of 2D slices
        number_of_time_frame: number of time frames
        Evenly_spaced_points: 3D coordinates of the contour points  
    """ 

    def __init__(self, filename, filenameInfo,case,time_frame_number):
        """ Return a DataSet object. Each point of this dataset is characterized by its 3D coordinates ([Evenly_spaced_points[:,0],Evenly_spaced_points[:,1],Evenly_spaced_points[:,2]]), the slice it belongs to (slice_number) and the surface its belongs to (ContourType)

            Input:
                filename: filename is the file containing the 3D contour points coordinates, labels and time frame (see example GPfile.txt).
                filenameInfo: filename is the file containing dicom info  (see example SliceInfoFile.txt).
                case: case number
                time_frame_number: time frame #
        """

        # column num 3 of my datset is a space (I don't know why but this is the output from the extraction code we have). You might need to change usecols=[0,1,2,4,5,6] to 
        # usecols=[0,1,2,3,4,5] for your dataset
        data = pd.read_csv(open(filename), sep='\t', header=None, usecols=[0,1,2,3,4])          
        P = data[[0,1,2]].values
        slices = data[4]
        self.DICOMslices = data[4]
        contypes = data[3]
        self.case = case
        #time_frame = data[6] 

        self.number_of_slice = max(slices)          
        #self.number_of_time_frame = max(time_frame) 

        #P = P[(time_frame == time_frame_number),:]
        #slices = slices[(time_frame == time_frame_number)]
        #self.DICOMslices = self.DICOMslices[(time_frame == time_frame_number)]
        #contypes = contypes[(time_frame == time_frame_number)]
               
        # calc valve centroids
        self.Mitral_centroid = P[(contypes == "BP_point"),:].mean(axis=0)
        self.Tricuspid_centroid = P[(contypes == "Tricuspid_Valve"),:].mean(axis=0) 
        self.Aorta_centroid = P[(contypes == "Aorta"),:].mean(axis=0)
        self.Pulmonary_centroid = P[(contypes == "Pulmonary"),:].mean(axis=0) 

        self.Apex= P[(contypes == "LA_Apex_Point"),:]

        if len(self.Apex) > 0:
            self.Apex = self.Apex[0,:]

        self.Evenly_spaced_points = P[(contypes == "BP_point"),:]
        self.slice_number = [-1]*len(P[(contypes == "BP_point"),:])
        self.ContourType = ["BP_point"]*len(P[(contypes == "BP_point"),:])

        self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,P[(contypes == "Tricuspid_Valve"),:]))
        self.slice_number = np.hstack((self.slice_number, slices[(contypes == "Tricuspid_Valve")]))
        self.ContourType = np.hstack((self.ContourType,["Tricuspid_Valve"]*len(P[(contypes == "Tricuspid_Valve"),:])))

        self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,P[(contypes == "Aorta"),:]))
        self.slice_number = np.hstack((self.slice_number, slices[(contypes == "Aorta")]))
        self.ContourType = np.hstack((self.ContourType,["Aorta"]*len(P[(contypes == "Aorta"),:])))

        self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,P[(contypes == "Pulmonary"),:]))
        self.slice_number = np.hstack((self.slice_number, slices[(contypes == "Pulmonary")]))
        self.ContourType = np.hstack((self.ContourType,["Pulmonary"]*len(P[(contypes == "Pulmonary"),:])))


        self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,P[(contypes == "LA_Apex_Point"),:]))
        self.slice_number = np.hstack((self.slice_number,[-1]*len(P[(contypes == "LA_Apex_Point"),:])))
        self.ContourType = np.hstack((self.ContourType,["LA_Apex_Point"]*len(P[(contypes == "LA_Apex_Point"),:])))        

        self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,P[(contypes == "RV_insert"),:]))
        self.slice_number = np.hstack((self.slice_number,slices[(contypes == "RV_insert")]))
        self.ContourType = np.hstack((self.ContourType,["RV_insert"]*len(P[(contypes == "RV_insert"),:]) )) 

        # Sampling of the contour points (we have too many 3D points and most of them cannot be projected onto the RVLV model as we are limited by the number 
        # of surface points of our model (5810 vertices))
        sample = 1
        for j in range(self.number_of_slice+1):  # For slice i, extract evenly spaced point for all type

            # LV LAX
            C = P[(contypes == "laendocardialContour") & (slices == j),:]

            # iterate through points until all points are taken away
            if len(C)>0:
                Cx = C[0,:]
                lastP = Cx
                Cr = np.delete(C, 0, 0)
                while Cr.shape[0]>0:

                    # find the closest point from the last point at Cx
                    i = (np.square(lastP - Cr)).sum(1).argmin()

                    # remove that closest point from Cr and add to Cx
                    lastP = Cr[i,:]
                    Cx = np.vstack([Cx, lastP])
                    Cr = np.delete(Cr, i, 0)

                    # -- END of while Cr.shape[0]>0            
                self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,Cx[0::sample,:]))
                self.slice_number = np.hstack((self.slice_number,[j]*len(Cx[0::sample,:])))
                self.ContourType = np.hstack((self.ContourType,["laendocardialContour"]*len(Cx[0::sample,:])))     

            # RV LAX
            C = P[(contypes == "larvendocardialContour") & (slices == j),:]
            # iterate through points until all points are taken away

            if len(C)>0:
                Cx = C[0,:]
                lastP = Cx
                Cr = np.delete(C, 0, 0)
                while Cr.shape[0]>0:

                    # find the closest point from the last point at Cx
                    i = (np.square(lastP - Cr)).sum(1).argmin()

                    # remove that closest point from Cr and add to Cx
                    lastP = Cr[i,:]
                    Cx = np.vstack([Cx, lastP])
                    Cr = np.delete(Cr, i, 0)

                    # -- END of while Cr.shape[0]>0            
                self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,Cx[0::sample,:]))
                self.slice_number = np.hstack((self.slice_number,[j]*len(Cx[0::sample,:])))
                self.ContourType = np.hstack((self.ContourType,["larvendocardialContour"]*len(Cx[0::sample,:])))  

            # LV LAX epi
            C = P[(contypes == "laepicardialContour") & (slices == j),:]

            # iterate through points until all points are taken away
            if len(C)>0:
                Cx = C[0,:]
                lastP = Cx
                Cr = np.delete(C, 0, 0)
                while Cr.shape[0]>0:

                    # find the closest point from the last point at Cx
                    i = (np.square(lastP - Cr)).sum(1).argmin()

                    # remove that closest point from Cr and add to Cx
                    lastP = Cr[i,:]
                    Cx = np.vstack([Cx, lastP])
                    Cr = np.delete(Cr, i, 0)

                    # -- END of while Cr.shape[0]>0            
                self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,Cx[0::sample,:]))
                self.slice_number = np.hstack((self.slice_number,[j]*len(Cx[0::sample,:])))
                self.ContourType = np.hstack((self.ContourType,["laepicardialContour"]*len(Cx[0::sample,:])))  


            # LV SAX endo points
            C = P[(contypes == "saendocardialContour") & (slices == j),:]
            # iterate through points until all points are taken away

            if len(C)>0:
                Cx = C[0,:]
                lastP = Cx
                Cr = np.delete(C, 0, 0)
                while Cr.shape[0]>0:

                    # find the closest point from the last point at Cx
                    i = (np.square(lastP - Cr)).sum(1).argmin()

                    # remove that closest point from Cr and add to Cx
                    lastP = Cr[i,:]
                    Cx = np.vstack([Cx, lastP])
                    Cr = np.delete(Cr, i, 0)

                    # -- END of while Cr.shape[0]>0            
                
                self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,Cx[0::sample,:]))
                self.slice_number = np.hstack((self.slice_number,[j]*len(Cx[0::sample,:])))
                self.ContourType = np.hstack((self.ContourType,["saendocardialContour"]*len(Cx[0::sample,:])))            
                
            # LA LAX open endo points
            C = P[(contypes == "lalaContour") & (slices == j),:]
            # iterate through points until all points are taken away

            if len(C)>0:
                Cx = C[0,:]
                lastP = Cx
                Cr = np.delete(C, 0, 0)
                while Cr.shape[0]>0:

                    # find the closest point from the last point at Cx
                    i = (np.square(lastP - Cr)).sum(1).argmin()

                    # remove that closest point from Cr and add to Cx
                    lastP = Cr[i,:]
                    Cx = np.vstack([Cx, lastP])
                    Cr = np.delete(Cr, i, 0)

                    # -- END of while Cr.shape[0]>0   
                    
                self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,Cx[0::sample,:]))
                self.slice_number = np.hstack((self.slice_number,[j]*len(Cx[0::sample,:])))
                self.ContourType = np.hstack((self.ContourType,["lalaContour"]*len(Cx[0::sample,:]))) 
           

            # RV SAX endo contours
            C = P[(contypes == "sarvendocardialContour") & (slices == j),:]
            # iterate through points until all points are taken away

            if len(C)>0:
                Cx = C[0,:]
                lastP = Cx
                Cr = np.delete(C, 0, 0)
                while Cr.shape[0]>0:

                    # find the closest point from the last point at Cx
                    i = (np.square(lastP - Cr)).sum(1).argmin()

                    # remove that closest point from Cr and add to Cx
                    lastP = Cr[i,:]
                    Cx = np.vstack([Cx, lastP])
                    Cr = np.delete(Cr, i, 0)

                    # -- END of while Cr.shape[0]>0          
                self.points = np.vstack((self.Evenly_spaced_points,Cx[0::1,:]))
                self.sn = np.hstack((self.slice_number,[j]*len(Cx[0::1,:])))
                self.CT = np.hstack((self.ContourType,["sarv"]*len(Cx[0::1,:]))) 
                
                self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,Cx[0::sample,:]))
                self.slice_number = np.hstack((self.slice_number,[j]*len(Cx[0::sample,:])))
                self.ContourType = np.hstack((self.ContourType,["sarv"]*len(Cx[0::sample,:])))
                
            # RVFW points            
            C = P[(contypes == "RVFW") & (slices == j),:]
            # iterate through points until all points are taken away

            if len(C)>0:
                Cx = C[0,:]
                lastP = Cx
                Cr = np.delete(C, 0, 0)

                while Cr.shape[0]>0:

                    # find the closest point from the last point at Cx
                    i = (np.square(lastP - Cr)).sum(1).argmin()

                    # remove that closest point from Cr and add to Cx
                    lastP = Cr[i,:]
                    Cx = np.vstack([Cx, lastP])
                    Cr = np.delete(Cr, i, 0)

                    # -- END of while Cr.shape[0]>0          

                self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,Cx[0::sample,:]))
                self.slice_number = np.hstack((self.slice_number,[j]*len(Cx[0::sample,:])))
                self.ContourType = np.hstack((self.ContourType,["RVFW"]*len(Cx[0::sample,:]))) 
                             
            # RVS points            
            C = P[(contypes == "RVS") & (slices == j),:]
            # iterate through points until all points are taken away

            if len(C)>0:
                Cx = C[0,:]
                lastP = Cx
                Cr = np.delete(C, 0, 0)

                while Cr.shape[0]>0:


                    # find the closest point from the last point at Cx
                    i = (np.square(lastP - Cr)).sum(1).argmin()

                    # remove that closest point from Cr and add to Cx
                    lastP = Cr[i,:]
                    Cx = np.vstack([Cx, lastP])
                    Cr = np.delete(Cr, i, 0)

                    # -- END of while Cr.shape[0]>0          

                self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,Cx[0::sample,:]))
                self.slice_number = np.hstack((self.slice_number,[j]*len(Cx[0::sample,:])))
                self.ContourType = np.hstack((self.ContourType,["RVS"]*len(Cx[0::sample,:]))) 
                
            # LV epi point            
            C = P[(contypes == "saepicardialContour") & (slices == j),:]
            # iterate through points until all points are taken away

            if len(C)>0:
                Cx = C[0,:]
                lastP = Cx
                Cr = np.delete(C, 0, 0)
                while Cr.shape[0]>0:

                    # find the closest point from the last point at Cx
                    i = (np.square(lastP - Cr)).sum(1).argmin()

                    # remove that closest point from Cr and add to Cx
                    lastP = Cr[i,:]
                    Cx = np.vstack([Cx, lastP])
                    Cr = np.delete(Cr, i, 0)

                    # -- END of while Cr.shape[0]>0          

                self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,Cx[1::sample,:]))
                self.slice_number = np.hstack((self.slice_number,[j]*len(Cx[1::sample,:])))
                self.ContourType = np.hstack((self.ContourType,["saepicardialContour"]*len(Cx[1::sample,:]))) 

        if self.Evenly_spaced_points.size > 0:
    
            # Read contour file info
            [self.ImagePositionPatient, self.ImageOrientationPatient, self.slice_num, self.PixelSpacing, self.slice_num] = self.Read_Slice_Info_file(filenameInfo)

            self.IsData = 1
        else:
            self.IsData = 0

        # Mitral points extracted from Circle belong to the slice -1 (I don't know why...). To be able to apply a breath-hold misregistration correction, 
        # we need to find which long axis slice each BP point was extracted from to apply the correct shift. 
        # If you don't have this problem with your dataset, you can comment it.
        #self.Identify_BP_Points()
       
    def Create_RV_epi(self):
        """ This function generates phantom points for the RV epi. Epicardium of the RV free wall was not manually contoured in our dataset, but it is better to have them when customizing the surface mesh.
            RV epicardial phantom points are estimated by extending the RV endocardium contour points by a fixed distance (3mm from the literature). If your dataset contains RV epicardial point, you can comment this function
            Input:
                None
            Output:
                RV_epi: RV epicardial phantom points
        """

        # RV_wall_thickness: normal value from literature
        number_of_slice = self.number_of_slice
        RV_epi = []

        RV_thickness = 3.0
        
        for i in range(number_of_slice+1):

            # For each slice, find centroid cloud point RVFW
            # Get contour points
            contour_points_slice = self.Evenly_spaced_points[(self.ContourType == "RVFW") & (self.slice_number == i),:]
            contour_points_slice = np.concatenate((contour_points_slice,self.Evenly_spaced_points[(self.ContourType == "sarvendocardialContour") & (self.slice_number == i),:]),axis=0)                   
            if len(contour_points_slice) > 0:
                slice_centroid = contour_points_slice.mean(axis=0)
                for j in contour_points_slice:
                    # get direction
                    direction = j[0:3] - slice_centroid
                    direction = direction / np.linalg.norm(direction)
                    
                    # Move j along direction by RV_thickness
                    new_position = np.add(j[0:3],np.array([RV_thickness*direction[0],RV_thickness*direction[1],RV_thickness*direction[2]]))
                    RV_epi.append(np.asarray([new_position[0],new_position[1],new_position[2],12,i]))
        
        return np.asarray(RV_epi) 


    def Read_Slice_Info_file(self,name):
        """ This function reads the 'name' file containing dicom info (see example SliceInfo.txt).

            Input:
                name: file_name

            Output:
                ImagePositionPatient: ImagePositionPatient attribute (x, y, and z coordinates of the upper left hand corner of the image)
                ImageOrientationPatient: ImageOrientationPatient attribute (specifies the direction cosines of the first row and the first column with respect to the patient)
                slice_num: slice #
                PixelSpacing: distance between the center of each pixel
        """

        lines = []
        with open (name, 'rt') as in_file:
            for line in in_file:    
                lines.append(line)  
        
        number_of_slice = int(len(lines)/8) # number of lines
        ImagePositionPatient = []
        ImageOrientationPatient = []
        DicomName = []
        PixelSpacing = []
        SliceID = []
        index_slice_number = 1
        index_imPos = 3
        index_Name = 0
        index_imOr = 5
        index_slice_num = 1
        index_pixel_spacing = 7
        
        for i in range(number_of_slice):
            DicomName.append('.'.join(re.findall('[-+]?\d+\.\d+', lines[index_Name])))  # DICOM name
            index_Name = index_Name + 8

            SliceID.append(re.findall('\d+', lines[index_slice_number]))  # DICOM name
            index_slice_number = index_slice_number + 8

            ImagePositionPatient.append(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?',lines[index_imPos]))
            index_imPos = index_imPos + 8
            ImageOrientationPatient.append(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?',lines[index_imOr]))
            
            index_imOr = index_imOr + 8 
            index_slice_num = index_slice_num + 8
            PixelSpacing.append(re.findall('[-+]?\d+\.\d+', lines[index_pixel_spacing]))
            index_pixel_spacing = index_pixel_spacing + 8
       
        ImagePositionPatient = np.asarray(ImagePositionPatient) 
        ImageOrientationPatient = np.asarray(ImageOrientationPatient)

        DicomName = np.asarray(DicomName)
        PixelSpacing = np.asarray(PixelSpacing)              
        #slice_num = pd.Series.unique(self.DataSet.DICOMslices)
        
        sliceID = []
        for item in SliceID:
            for a in item:
                sliceID.append(int(a))
        slice_num = sliceID
        return [ImagePositionPatient, ImageOrientationPatient, slice_num, PixelSpacing, sliceID]
               

    def Identify_BP_Points(self):
        """ This function matches each BP_point with the LAX slice it was extracted from.
            Input:
                None
            Output:
                None. slice_number for each BP_point is changed to the corresponding LAX slice number
        """        

        BP_points = self.Evenly_spaced_points[(self.ContourType=="BP_point"),:]  # Extraction code extracts BP points' slice as -1        
        idx = np.where(self.ContourType=="BP_point")

        New_BP = np.zeros((len(BP_points),3))
        Corresponding_slice = []
        it = 0
        for slices in range(len(self.ImagePositionPatient)):

            LAX = self.Evenly_spaced_points[(self.ContourType=="lalaContour") & (self.slice_number == slices) ,:] 
            if len(LAX) > 0:
                minimum = 100
                Corr = np.zeros((len(BP_points),1))
                NP = np.zeros((len(BP_points),3))
                Sl = np.zeros((len(BP_points),1))
                # Find corresponding BP on this slices - should be two

                for points in range(len(BP_points)):
                    i = (np.square(BP_points[points,:] - LAX)).sum(1).argmin()    
                    Corr[points] = np.linalg.norm(LAX[i,:]-BP_points[points,:])
                    NP[points] = LAX[i,:]
                    Sl[points] = slices


                index = Corr.argmin()
                New_BP[it,:] = NP[index,:]
                Corresponding_slice.append(float(Sl[index]))

                NP = np.delete(NP,index, 0)
                Sl = np.delete(Sl,index, 0)
                Corr = np.delete(Corr,index,0)
                it = it+1

                index = Corr.argmin()
                New_BP[it,:] = NP[index,:]
                Corresponding_slice.append(float(Sl[index]))
                it = it+1

        indexes = np.where((self.ContourType=="BP_point"))

        self.Evenly_spaced_points = np.delete(self.Evenly_spaced_points, indexes, 0)               
        self.ContourType = np.delete(self.ContourType, indexes, 0)       
        self.slice_number = np.delete(self.slice_number, indexes, 0)  

        self.Evenly_spaced_points = np.vstack((self.Evenly_spaced_points,New_BP))
        self.ContourType = np.hstack((self.ContourType,["BP_point"]*len(New_BP))) 
        self.slice_number = np.hstack((self.slice_number,Corresponding_slice))


    def Identify_RVS_LAX(self):
        """ This function splits RV LAX contours (labelled as larvendocardialContour) into RVS and larvendocardialContour.
            Input:
                None
            Output:
                None. ContourType for each larvendocardialContour point is changed to the corresponding label (larvendocardialContour or RVS)
        """  

        for slices in range(len(self.ImagePositionPatient)):

            RVLAX = self.Evenly_spaced_points[(self.ContourType=="larvendocardialContour") & (self.slice_number == slices) ,:] 
            
            if len(RVLAX) > 0:
                LVLAX = self.Evenly_spaced_points[(self.ContourType=="laepicardialContour") & (self.slice_number == slices) ,:] 
                if len(LVLAX) > 0:
 
                    # Find matching points
                    indxRV =  np.where((self.ContourType=="larvendocardialContour") & (self.slice_number == slices))[0]                
                    indexes = np.where((self.ContourType=="laepicardialContour") & (self.slice_number == slices))[0]

                    for i in range(0,len(RVLAX)):
                        query = RVLAX[i,:]
                        distance, idx = spatial.KDTree(LVLAX).query(query)

                        if distance < 3:
                            self.ContourType[indxRV[i]] = "RVS"

                            self.Evenly_spaced_points = np.delete(self.Evenly_spaced_points, indexes[idx], 0)               
                            self.ContourType = np.delete(self.ContourType, indexes[idx], 0)       
                            self.slice_number = np.delete(self.slice_number, indexes[idx], 0)   


                    # Get RV insertion points - farthest points
                    X = self.Evenly_spaced_points[(self.ContourType=="RVS") & (self.slice_number == slices) ,:] 
                    indxRV =  np.where((self.ContourType=="RVS") & (self.slice_number == slices))[0] 

                    dist_mat = spatial.distance_matrix(X,X)

                    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

                    # I am not sure if the selected point next to the tricuspid valve can be defined as RV insert so I just select the one near the apex just in case.
                    if (np.linalg.norm(X[i,:]-self.Tricuspid_centroid) > np.linalg.norm(X[j,:]-self.Tricuspid_centroid)):
                        self.ContourType[indxRV[i]] = "RV_insert"
                    else:
                        self.ContourType[indxRV[j]] = "RV_insert"

    def Create_Mitral_phantomPoints(self,n):
        """ This function creates mitral phantom points by fitting a circle to the mitral points from the DataSet 
            Input:
                n: number of phantom points we want to create
            Output:
                P_fitcircle: phantom points
        """  

        BP_points = self.Evenly_spaced_points[self.ContourType == "BP_point",:] 

        # Coordinates of the 3D points
        P = np.array(BP_points) 

        distance1 = np.linalg.norm(P[0,:] - P[1,:])
        distance2 = np.linalg.norm(P[2,:] - P[3,:])

        if distance2 < 20 and distance1 > 20:

            C =  0.5*P[0,:] + 0.5*P[1,:]        

            u = P[0,:] - C
            u = u/np.linalg.norm(u) 
            normal = self.Evenly_spaced_points[self.ContourType == "saendocardialContour",:].mean(axis=0) - C
            normal = normal/np.linalg.norm(normal)

            r = distance1/2

            t = np.linspace(-np.pi, np.pi, n)
            P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)

        elif distance1 < 20 and distance2 > 20:
            C =  0.5*P[2,:] + 0.5*P[3,:] 

            u = P[2,:] - C
            u = u/np.linalg.norm(u) 

            normal = self.Evenly_spaced_points[self.ContourType == "saendocardialContour",:].mean(axis=0) - C
            normal = normal/np.linalg.norm(normal)

            r = distance2/2
           
            t = np.linspace(-np.pi, np.pi, n)
            P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)

        else:
            P_mean = P.mean(axis=0)
            P_centered = P - P_mean
            U,s,V = np.linalg.svd(P_centered)

            # Normal vector of fitting plane is given by 3rd column in V
            # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
            normal = V[2,:]
            d = -np.dot(P_mean, normal)  # d = -<p,n>

            #-------------------------------------------------------------------------------
            # (2) Project points to coords X-Y in 2D plane
            #-------------------------------------------------------------------------------
            P_xy = rodrigues_rot(P_centered, normal, [0,0,1])
            xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])
            C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + P_mean
            C = C.flatten()
    
            #--- Generate points for fitting circle
            t = np.linspace(-np.pi, np.pi, n)
            u = P[0] - C

            P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)

        return P_fitcircle

    def Create_Tricuspid_phantom_Points_Ellipse(self, n):
        """ This function creates tricuspid phantom points by fitting a circle to the tricuspid points from the DataSet 
            Input:
                n: number of phantom points we want to create
            Output:
                P_fitcircle: phantom points
        """ 

        Tricuspid_points = self.Evenly_spaced_points[self.ContourType == "Tricuspid_Valve",:] 

        # Coordinates of the 3D points
        P = np.array(Tricuspid_points) 

        P_mean = P.mean(axis=0)
        P_centered = P - P_mean
        U,s,V = np.linalg.svd(P_centered)

        normal = V[2,:]

        # create direction vectors
        v1 = np.array([-normal[1],normal[0],0])
        v2 = np.array([-normal[2],0,normal[0]])

        # Translation
        T = np.identity(4)
        T[0:3,3] = P_mean

        # Rotation
        R = np.identity(4)
        R[0:3,0] = v1
        R[0:3,1] = v2
        R[0:3,2] = normal

        Transformation = np.dot(T,R)

        Transformation2D = np.linalg.inv(Transformation) 
        pts_LV = np.ones((len(P),4))
        pts_LV[:,0:3] = P
        Px_LV = np.dot(pts_LV,Transformation2D.T)
        P_xy = Px_LV[:,0:2] / (np.vstack((Px_LV[:,3],Px_LV[:,3]))).T

        #print(P_xy)
        #d = -np.dot(P_mean, normal)

        #P_xy = rodrigues_rot(P_centered, normal, [0,0,1])
        #print(P_xy)

        # Extract x coords and y coords of the ellipse as column vectors
        X = P_xy[:,0]
        Y = P_xy[:,1]

        # Formulate and solve the least squares problem ||Ax - b ||^2
        A = np.vstack([X**2, X * Y, Y**2, X, Y])
        b = np.ones_like(X)

        x = np.linalg.lstsq(A.T, b)[0].squeeze()

        # Print the equation of the ellipse in standard form
        print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1],x[2],x[3],x[4]))

        phi = np.linspace(0, 2*np.pi, 1000).reshape((1000,1))
        c = np.hstack([np.cos(phi), np.sin(phi)])

        x_coord = np.linspace(-100,100,n)
        y_coord = np.linspace(-100,100,n)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
        CS = plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)

        dat0 = CS.allsegs[0][0]

        pts_LV = np.ones((len(dat0),4))
        pts_LV[:,0:2] = dat0
        pts_LV[:,2] = [0]*len(dat0)
        pts_LV[:,3] = [1]*len(dat0)

        Px_LV = np.dot(Transformation,pts_LV.T)
        P3_LV = Px_LV[0:3,:] / (np.vstack((Px_LV[3,:],np.vstack((Px_LV[3,:],Px_LV[3,:])))))

        return P3_LV.T

    def Create_Tricuspid_phantomPoints(self, n):
        """ This function creates tricuspid phantom points by fitting a circle to the tricuspid points from the DataSet 
            Input:
                n: number of phantom points we want to create
            Output:
                P_fitcircle: phantom points
        """ 

        Tricuspid_points = self.Evenly_spaced_points[self.ContourType == "Tricuspid_Valve",:] 

        # Coordinates of the 3D points
        P = np.array(Tricuspid_points) 


        if len(P) > 2:
            P_mean = P.mean(axis=0)
            P_centered = P - P_mean
            U,s,V = np.linalg.svd(P_centered)

            normal = V[2,:]
            d = -np.dot(P_mean, normal)

            P_xy = rodrigues_rot(P_centered, normal, [0,0,1])
            xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])

            C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + P_mean
            C = C.flatten()

            t = np.linspace(-np.pi, np.pi, n)
            u = P[0] - C

            P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)

        else:
            distance = np.linalg.norm(P[0,:] - P[1,:])

            C =  0.5*P[0,:] + 0.5*P[1,:]        

            u = P[0,:] - C
            u = u/np.linalg.norm(u) 
            normal = self.Evenly_spaced_points[self.ContourType == "larvendocardialContour",:].mean(axis=0) - C
            normal = normal/np.linalg.norm(normal)

            r = distance/2

            t = np.linspace(-np.pi, np.pi, n)

            normal = normal/np.linalg.norm(normal)
            u = u/np.linalg.norm(u)
            P_fitcircle = r*np.cos(t)[:,np.newaxis]*u + r*np.sin(t)[:,np.newaxis]*np.cross(normal,u) + C

        return P_fitcircle

    def Create_Aorta_phantomPoints(self, n):
        """ This function creates tricuspid phantom points by fitting a circle to the tricuspid points from the DataSet 
            Input:
                n: number of phantom points we want to create
            Output:
                P_fitcircle: phantom points
        """ 

        Aorta_points = self.Evenly_spaced_points[self.ContourType == "Aorta",:] 

        # Coordinates of the 3D points
        P = np.array(Aorta_points) 

        if len(P) <3:
            distance = np.linalg.norm(P[0,:] - P[1,:])

            C =  0.5*P[0,:] + 0.5*P[1,:]        

            u = P[0,:] - C
            u = u/np.linalg.norm(u) 
            normal = self.Evenly_spaced_points[self.ContourType == "saendocardialContour",:].mean(axis=0) - C
            normal = normal/np.linalg.norm(normal)

            r = distance/2

            t = np.linspace(-np.pi, np.pi, n)

            normal = normal/np.linalg.norm(normal)
            u = u/np.linalg.norm(u)
            P_fitcircle = r*np.cos(t)[:,np.newaxis]*u + r*np.sin(t)[:,np.newaxis]*np.cross(normal,u) + C

        else: # if more than 2 points, we have enough to define a circle
            P_mean = P.mean(axis=0)
            P_centered = P - P_mean
            U,s,V = np.linalg.svd(P_centered)

            normal = V[2,:]
            d = -np.dot(P_mean, normal)

            P_xy = rodrigues_rot(P_centered, normal, [0,0,1])
            xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])
            C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + P_mean
            C = C.flatten()

            t = np.linspace(-np.pi, np.pi, n)
            u = P[0] - C

            P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)

        return P_fitcircle

    def Create_Pulmonary_phantomPoints(self, n):
        """ This function creates tricuspid phantom points by fitting a circle to the tricuspid points from the DataSet 
            Input:
                n: number of phantom points we want to create
            Output:
                P_fitcircle: phantom points
        """ 

        Pulmonary_points = self.Evenly_spaced_points[self.ContourType == "Pulmonary",:] 

        # Coordinates of the 3D points
        P = np.array(Pulmonary_points) 

        if len(P) > 2:

            P_mean = P.mean(axis=0)
            P_centered = P - P_mean
            U,s,V = np.linalg.svd(P_centered)

            normal = V[2,:]
            d = -np.dot(P_mean, normal)

            P_xy = rodrigues_rot(P_centered, normal, [0,0,1])
            xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])
            C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + P_mean
            C = C.flatten()

            t = np.linspace(-np.pi, np.pi, n)
            u = P[0] - C

            P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)

        else:
            distance = np.linalg.norm(P[0,:] - P[1,:])

            C =  0.5*P[0,:] + 0.5*P[1,:]        

            u = P[0,:] - C
            u = u/np.linalg.norm(u) 
            normal = self.Evenly_spaced_points[self.ContourType == "RVFW",:].mean(axis=0) - C
            normal = normal/np.linalg.norm(normal)

            r = distance/2

            t = np.linspace(-np.pi, np.pi, n)

            normal = normal/np.linalg.norm(normal)
            u = u/np.linalg.norm(u)
            P_fitcircle = r*np.cos(t)[:,np.newaxis]*u + r*np.sin(t)[:,np.newaxis]*np.cross(normal,u) + C

        return P_fitcircle

    def From2Dto3D(self,ImagePositionPatient,ImageOrientationPatient,slice_num):
        """ This function calculates transformation matrix needed to convert 2D points to 3D points, given ImagePositionPatient and ImageOrientationPatient. 

            Input:
                ImagePositionPatient: ImagePositionPatient attribute (x, y, and z coordinates of the upper left hand corner of the image)
                ImageOrientationPatient: ImageOrientationPatient attribute (specifies the direction cosines of the first row and the first column with respect to the patient)
                slice_num: slice #
              
            Output:
                T: transformation matrix
        """         
        Spacing = self.PixelSpacing[slice_num,:]            

        # Translation
        T = np.identity(4)
        T[0:3,3] = ImagePositionPatient
        
        # Rotation
        R = np.identity(4)
        R[0:3,0] = ImageOrientationPatient[0:3]
        R[0:3,1] = ImageOrientationPatient[3:7]
        R[0:3,2] = np.cross(R[0:3,0],R[0:3,1])
        
        # scale
        S = np.identity(4)
        S[0,0] = Spacing[1]
        S[1,1] = Spacing[0]
        
        T = np.dot(T,R)
        T = np.dot(T,S)
        
        return T


    def PlotDataSetEvenlySpaced(self):
        """ This function plots this entire dataset.
            Input:
                None
            Output:
                traces for figure
        """  

        Tricuspid_phantom = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == "Tricuspid_phantom")],"rgb(128,0,128)", 2,'Tricuspid_phantom')
        Tricuspid = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == "Tricuspid_Valve")],"rgb(128,0,128)", 5,'Tricuspid')
        Pulmonary_phantom = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == "Pulmonary_phantom")],"rgb(0,43,0)", 2,'Pulmonary_phantom')
        Pulmonary = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == "Pulmonary")],"rgb(0,43,0)", 5,'Pulmonary')
        Aorta_phantom = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == "Aorta_phantom")],"rgb(0,255,0)", 2,'Aorta_phantom')
        Aorta = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == "Aorta")],"rgb(0,255,0)", 5,'Aorta')
        Mitral_phantom = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == "BP_phantom")],"rgb(255,0,0)", 2,'Mitral_phantom')
        Mitral = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == "BP_point")],"rgb(255,0,0)", 5,'Mitral')
        RV = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == ("RVFW" or "larvendocardialContour" or "sarvendocardialContour"))],"rgb(0,0,205)", 2,'RVFW')
        RVS = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == "RVS")],"rgb(139,0,139)", 2,'RVS')
        LV = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == ("saendocardialContour" or "laendocardialContour"))],"rgb(85,107,47)", 2,'LV')
        Epi = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == ("saepicardialContour" or "laepicardialContour" or "RVepicardialContour" or "larvepicardialContour"))],"rgb(220,20,60)", 2,'Epicardium')
        RVinsert = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == "RV_insert")],"rgb(255,20,147)", 5,'RV_insert')
        Apex = Plot3DPoint(self.Evenly_spaced_points[np.where(np.asarray(self.ContourType) == "LA_Apex_Point")],"rgb(0,191,255)", 5,'Apex')
        
        return [Mitral_phantom, Mitral , Tricuspid_phantom, Tricuspid, Pulmonary_phantom, Pulmonary, Aorta_phantom, Aorta, RV, RVS,LV, Epi, RVinsert, Apex]
