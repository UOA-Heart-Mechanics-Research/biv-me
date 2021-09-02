# -*- coding: utf-8 -*-
import numpy as np
import sys
from scipy import spatial
sys.path.append('C:/Users/cmau619/AppData/Local/Programs/Python/Python35-32/Lib/site-packages')
import time
from scipy.spatial.distance import cdist
import functools
import GPDataSet
import importlib
importlib.reload(GPDataSet)
from GPDataSet import*
import copy
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.interpolate import SmoothBivariateSpline
from scipy.interpolate import splprep
from scipy.interpolate import splev
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import directed_hausdorff
import scipy.optimize
import pandas as pd
import numba
#import cvxopt
from cvxopt import matrix, solvers

##Author : Charlène Mauger, University of Auckland, c.mauger@auckland.ac.nz
class biventricularModel(object):
    """ This class creates a surface from the control mesh, based on Catmull-Clark subdivision surface method. Surfaces have the following properties:

    Attributes:
       etPos                                Array of x,y,z coordinates for each surface point (5810x3).                                         
       etVertexElementNum                   Element num for each surface point (5810x1).  
       matrix                               Subdivision matrix (388x5810).                                
       control_mesh                         Control mesh (388x3).  
       ETIndices                            Elements connectivities (n1, n2, n3) for each face (11760x3).  
       GTSTSG_x, GTSTSG_y, GTSTSG_z         Regularization/Smoothing matrices (388x388) along Xi1 (circumferential), Xi2 (transmural) and Xi3 (transmural) directions 
       numNodes = 388                       Number of control nodes.                
       numElements = 187                    Number of elements.
       numSurfaceNodes = 5810               Number of nodes after subdivision (surface points).       
       Apex_index                           Vertex index of the apex          
       basis_matrix                         Matrix (5810x388) containing basis functions used to evaluate surface at surface point locations
       etVertexStartEnd                     Surface index limits for vertices etPos. Surfaces are sorted in the following order:  LV, RV septum, RV free wall, epicardium, mitral valve, aorta, tricuspid, pulmonary valve, RV insert. Valve centroids are always the last vertex of the corresponding surface
       SurfaceStartEnd                      Surface index limits for embedded triangles ETIndices. Surfaces are sorted in the following order:  LV, RV septum, RV free wall, epicardium, mitral valve, aorta, tricuspid, pulmonary valve, RV insert.
       mBder_dx, mBder_dy, mBder_dz         Matrices (5049x338) containing weights used to calculate gradients of the displacement field at Gauss point locations.
       Jac11, Jac12, Jac13                  Matrices (11968x388) containing weights used to calculate Jacobians at Gauss point location (11968x338). Each matrix element is a linear combination of the 388 control points. 
                                            J11 contains the weight used to compute the derivatives along Xi1, J12 along Xi2 and J13 along Xi3.  Jacobian determinant is calculated/checked on 11968 locations.              
    """

    numNodes = 388  
    numElements = 187
    numSurfaceNodes = 5810
         
    def __init__(self, control_mesh, DataSet,case,FileInfo):
        """ Return a Surface object whose control mesh is *control_mesh* and should be fitted to the dataset *DataSet*
            control_mesh is always the same - this is the RVLV template. If you change the template, you need to regenerate all the matrices. 
        """
        self.case = case
        self.control_mesh = control_mesh
        self.matrix = (pd.read_table("subdivision_matrix.txt",delim_whitespace=True,header=None)).values
        self.etPos = np.dot(self.matrix,control_mesh) 
        self.ETIndices = (pd.read_table("ETIndicesSorted.txt",delim_whitespace=True,header=None)).values

        self.GTSTSG_x = (pd.read_table("GTSTG_x.txt",delim_whitespace=True,header=None)).values
        self.GTSTSG_y = (pd.read_table("GTSTG_y.txt",delim_whitespace=True,header=None)).values
        self.GTSTSG_z = (pd.read_table("GTSTG_z.txt",delim_whitespace=True,header=None)).values

        self.etVertexElementNum = (pd.read_table("etVertexElementNum.txt",delim_whitespace=True,header=None)).values 

        self.mBder_dx = (pd.read_table("mBder_x.txt",delim_whitespace=True,header=None)).values
        self.mBder_dy = (pd.read_table("mBder_y.txt",delim_whitespace=True,header=None)).values
        self.mBder_dz = (pd.read_table("mBder_z.txt",delim_whitespace=True,header=None)).values 

        self.Jac11 = (pd.read_table("J11.txt",delim_whitespace=True,header=None)).values 
        self.Jac12 = (pd.read_table("J12.txt",delim_whitespace=True,header=None)).values
        self.Jac13 = (pd.read_table("J13.txt",delim_whitespace=True,header=None)).values      
                
        self.basis_matrix = (pd.read_table("basis_matrix.txt",delim_whitespace=True,header=None)).values # OK
        self.Apex_index = 50#50# endo #5485 #epi
        self.etVertexStartEnd=np.array([[0,1499],[1500,2164],[2165,3223],[3224,5581],[5582,5630],[5631,5655],[5656,5696],[5697,5729],[5730,5809]]) 
        self.SurfaceStartEnd=np.array([[0,3071],[3072,4479],[4480,6751],[6752,11615],[11616,1163],[11664,11687],[11688,11727],[11728,11759]])

        self.DataSet = DataSet    

        [self.ImagePositionPatient, self.ImageOrientationPatient, self.slice_num, self.PixelSpacing] = self.Read_Slice_Info_file(FileInfo)


    def GetSurfaceStartEnd(self, surface_name):
        """ Get Surface Start and End 
                Input: 
                    surface_name: surface name
                Output: 
                    array containing first and last vertices index belonging to surface_name
        """

        if surface_name == "LV":            
            return self.etVertexStartEnd[0,:]           

        if surface_name == "RVS":            
            return self.etVertexStartEnd[1,:]  
            
        if surface_name == "RVFW":            
            return self.etVertexStartEnd[2,:] 
            
        if surface_name == "Epi": 
            return self.etVertexStartEnd[3,:]
            
        if surface_name == "Mitral":          
            return self.etVertexStartEnd[4,:]           

        if surface_name == "Aorta":            
            return self.etVertexStartEnd[5,:]  
            
        if surface_name == "Tricuspid":            
            return self.etVertexStartEnd[6,:] 
            
        if surface_name == "Pulmonary": 
            return self.etVertexStartEnd[7,:] 

        if surface_name == "RV_insert": 
            return self.etVertexStartEnd[8,:]  

    def IsDiffeomorphic(self,new_control_mesh,min_jacobian):
        """ This function checks the Jacobian value at Gauss point location (I am using 3x3x3 per element). 
            This function returns 0 if one of the determinants is below a given threshold and 1 otherwise. 
            I usually use 0.1 to make sure that there is no intersection/folding (We can also use 0, but it might 
            still give a positive jacobian if there are small intersections due to numerical approximation.
            Input: 
                new_control_mesh: control mesh we want to check 
            Output:
                min_jacobian: Jacobian threshold
            """

        boolean = 1
        for i in range(len(self.Jac11)):
            jacobi = np.array([[np.inner(self.Jac11[i,:],new_control_mesh[:,0]),np.inner(self.Jac12[i,:],new_control_mesh[:,0]),np.inner(self.Jac13[i,:],new_control_mesh[:,0])],
                               [np.inner(self.Jac11[i,:],new_control_mesh[:,1]),np.inner(self.Jac12[i,:],new_control_mesh[:,1]),np.inner(self.Jac13[i,:],new_control_mesh[:,1])],
                               [np.inner(self.Jac11[i,:],new_control_mesh[:,2]),np.inner(self.Jac12[i,:],new_control_mesh[:,2]),np.inner(self.Jac13[i,:],new_control_mesh[:,2])]])                         
            determinant = np.linalg.det(jacobi)
            
            if determinant < min_jacobian:
                boolean = 0
                return boolean 
                
        return boolean 
                                                                                
    def CalcDataSetXi(self,Weight):
        """ This function calculates the data basis function matrix. It projects the N data points onto the closest point in the
            corresponding model surface. If 2 data points are projected onto the same surface point, the closest one is kept.
                Input: 
                    Weight: weight given to the N data points

                Output:
                    Psi_matrix: basis function matrix (Nx388)
                    index: closest points indices (Nx1)
                    d: data points (Nx3)
                    W: Weight on the data points (N*N). Higher weights are given to RV_insert and valve points.
                    ContourType: 0 if this point belongs to the LV, 1 for RV. This is to get RV and LV error if needed (Nx1)
                    distance_d_prior: distances to the closest points (Nx1)
        """

        Sampled_points = self.DataSet.Evenly_spaced_points
        typeP = self.DataSet.ContourType

        Psi_matrix = []
        d = []
        W = []
        ContourType =  []
        distance_d_prior = []
        index = []
        d = []

        # Trees initialization
        basis_matrix = self.basis_matrix
        treeLV = scipy.spatial.cKDTree(self.etPos[self.etVertexStartEnd[0,0]:self.etVertexStartEnd[0,1],:])
        treeRVFW = scipy.spatial.cKDTree(self.etPos[self.etVertexStartEnd[2,0]:self.etVertexStartEnd[2,1],:])
        treeRVS = scipy.spatial.cKDTree(self.etPos[self.etVertexStartEnd[1,0]:self.etVertexStartEnd[1,1],:])
        treeTr = scipy.spatial.cKDTree(self.etPos[self.etVertexStartEnd[6,0]:self.etVertexStartEnd[6,1],:])
        treeEpi = scipy.spatial.cKDTree(self.etPos[self.etVertexStartEnd[3,0]:self.etVertexStartEnd[3,1],:])
        treeBP = scipy.spatial.cKDTree(self.etPos[self.etVertexStartEnd[4,0]:self.etVertexStartEnd[4,1],:])
        treeAo = scipy.spatial.cKDTree(self.etPos[self.etVertexStartEnd[5,0]:self.etVertexStartEnd[5,1],:])
        treePu = scipy.spatial.cKDTree(self.etPos[self.etVertexStartEnd[7,0]:self.etVertexStartEnd[7,1],:])
        treeRVInsert = scipy.spatial.cKDTree(self.etPos[self.etVertexStartEnd[8,0]:self.etVertexStartEnd[8,1],:])

        for zd in range(len(Sampled_points)):
            typePoint = typeP[zd]
            query = Sampled_points[zd,0:3]
            
            if typePoint == "saendocardialContour" or typePoint == "laendocardialContour": # LV endo (sa: short axis, la: long axis)
                distance, idx = treeLV.query(query, k=1, p=2)
                index_closest = idx + self.etVertexStartEnd[0,0]
                
                if index_closest not in index:
                    index.append(index_closest)
                    Psi_matrix.append(basis_matrix[int(index_closest),:])
                    W.append(Weight)
                    #ContourType.append(0)
                    distance_d_prior.append(distance)
                    d.append(Sampled_points[zd,0:3])
                else:
                    old_idx = index.index(index_closest)
                    distance_old = distance_d_prior[old_idx]
                    if distance < distance_old:
                        distance_d_prior[old_idx] = distance                                               
                        d[old_idx] = Sampled_points[zd,0:3]           
                        
            if (typePoint == 'RVFW' or typePoint == 'sarvendocardialContour' or typePoint == 'larvendocardialContour'): # It's RV free wall 
                distance, idx = treeRVFW.query(query, k=1, p=2)
                index_closest = idx + self.etVertexStartEnd[2,0]

                if index_closest not in index:
                    index.append(index_closest)
                    Psi_matrix.append(basis_matrix[int(index_closest),:])
                    W.append(Weight)
                    distance_d_prior.append(distance)
                    d.append(Sampled_points[zd,0:3])
                else:
                    old_idx = index.index(index_closest)
                    distance_old = distance_d_prior[old_idx]
                    if distance < distance_old:
                        distance_d_prior[old_idx] = distance                                               
                        d[old_idx] = Sampled_points[zd,0:3]

            if typePoint == "RVS": # RV septum 
                distance, idx = treeRVS.query(query, k=1, p=2)
                index_closest = idx + self.etVertexStartEnd[1,0]
                
                if index_closest not in index:
                    index.append(index_closest)
                    Psi_matrix.append(basis_matrix[int(index_closest),:])
                    W.append(Weight)
                    distance_d_prior.append(distance)
                    d.append(Sampled_points[zd,0:3])
                else:
                    old_idx = index.index(index_closest)
                    distance_old = distance_d_prior[old_idx]
                    if distance < distance_old:
                        distance_d_prior[old_idx] = distance                                               
                        d[old_idx] = Sampled_points[zd,0:3]

                
            if (typePoint == 'saepicardialContour' or typePoint == 'laepicardialContour' or typePoint == 'RVepicardialContour'): # Epicardium (SA: short axis, LA: long axis)
                distance, idx = treeEpi.query(query, k=1, p=2)
                index_closest = idx + self.etVertexStartEnd[3,0]
                
                if index_closest not in index:
                    index.append(index_closest)
                    Psi_matrix.append(basis_matrix[int(index_closest),:])
                    W.append(Weight)
                    #ContourType.append(0)
                    distance_d_prior.append(distance)
                    d.append(Sampled_points[zd,0:3])
                else:
                    old_idx = index.index(index_closest)
                    distance_old = distance_d_prior[old_idx]
                    if distance < distance_old:
                        distance_d_prior[old_idx] = distance                                               
                        d[old_idx] = Sampled_points[zd,0:3]
                          
            # If it is a valve, we virtually translate the data points (only the ones belonging to the same surface) so their centroid matches the template's valve centroid. 
            # So instead of calculating the minimum distance between the point p and the model points pm, we calculate the minimum distance between the point p+t and pm, 
            # where t is the translation needed to match both centroids 
            # This is to make sure that the data points are going to be projected all around the valve and not only on one side. 
            if typePoint == "BP_point" or typePoint == "BP_phantom": # BP points (=mitral points)

                centroid_valve = self.etPos[self.etVertexStartEnd[4,1],:]
                centroid_GP_mitral = self.DataSet.Mitral_centroid
                translation_GP_model = centroid_valve - centroid_GP_mitral
                d.append(Sampled_points[zd,0:3]) 
                distance, idx = treeBP.query(np.add(query,translation_GP_model), k=1, p=2)
                index_closest = idx + self.etVertexStartEnd[4,0]
                index.append(index_closest)
                Psi_matrix.append(basis_matrix[index_closest,:])
                W.append(5*Weight)
                distance_d_prior.append(distance) 
 
            if typePoint == "Aorta" or typePoint == "Aorta_phantom" : 
                centroid_valve = self.etPos[self.etVertexStartEnd[5,1],:]
                centroid_GP_aorta = self.DataSet.Aorta_centroid
                translation_GP_model = centroid_valve - centroid_GP_aorta
                d.append(Sampled_points[zd,0:3])    
                distance, idx = treeAo.query(np.add(query,translation_GP_model), k=1, p=2)
                index_closest = idx + self.etVertexStartEnd[5,0]
                index.append(index_closest)
                Psi_matrix.append(basis_matrix[index_closest,:])
                W.append(5*Weight)
                distance_d_prior.append(distance)

            if typePoint == "Tricuspid_Valve" or typePoint == "Tricuspid_phantom": # Tricuspid 
                centroid_valve = self.etPos[self.etVertexStartEnd[6,1],:]
                centroid_GP_tricuspid = self.DataSet.Tricuspid_centroid
                translation_GP_model = centroid_valve - centroid_GP_tricuspid                 
                distance, idx = treeTr.query(np.add(query,translation_GP_model), k=1, p=2)
                index_closest = idx + self.etVertexStartEnd[6,0]
                index.append(index_closest)
                Psi_matrix.append(basis_matrix[index_closest,:])
                W.append(5*Weight)
                distance_d_prior.append(distance)
                d.append(Sampled_points[zd,0:3])             

            if typePoint == "Pulmonary" or typePoint == "Pulmonary_phantom" : # Pulmonary 
                centroid_valve = self.etPos[self.etVertexStartEnd[7,1],:]
                centroid_GP_pulmonary = self.DataSet.Pulmonary_centroid
                translation_GP_model = centroid_valve - centroid_GP_pulmonary               
                distance, idx = treePu.query(np.add(query,translation_GP_model), k=1, p=2)

                index_closest = idx + self.etVertexStartEnd[7,0]
                index.append(index_closest)
                Psi_matrix.append(basis_matrix[index_closest,:])
                W.append(5*Weight)
                distance_d_prior.append(distance)
                d.append(Sampled_points[zd,0:3])
   
            if typePoint == "RV_insert": # RV insertion
                distance, idx = treeRVInsert.query(query, k=1, p=2)
                d.append(Sampled_points[zd,0:3])
                index_closest = idx + self.etVertexStartEnd[8,0]
                index.append(index_closest)
                Psi_matrix.append(basis_matrix[index_closest,:])
                W.append(5*Weight)
                distance_d_prior.append(distance)         
                
            if typePoint == "LA_Apex_Point": # Apex           
                distance = np.linalg.norm(self.etPos[self.Apex_index,:] - query)
                d.append(Sampled_points[zd,0:3])
                index_closest = self.Apex_index
                index.append(index_closest)
                Psi_matrix.append(basis_matrix[int(index_closest),:])
                W.append(5*Weight)
                distance_d_prior.append(distance)               

        return [np.asarray(Psi_matrix),np.asarray(index),np.asarray(d),np.asarray(W),np.asarray(distance_d_prior)]

    def LVLAXSAXSliceShifting(self):
        """ This method does a breath-hold misregistration correction for both LAX
            and SAX using Sinclair, Matthew, et al. "Fully automated segmentation-based 
            respiratory motion correction of multiplanar cardiac magnetic resonance images 
            for large-scale datasets." International Conference on Medical Image Computing 
            and Computer-Assisted Intervention. Springer, Cham, 2017. Briefly, this
            method iteratively registers each slice to its intersection with the other slices, which are kept fixed.
            This is performed at ED only. Translations from ED are applied to the others. Rotations and out-of-plane 
            displacements were assumed to be negligible relative to the slice thickness.

            Input: 
               None. Translations are done on the object 'self'.
            Output:
               2D translations needed (N*2, where N is the number of slices).
        """
        stoping_criterion = 5;  # The stoping_criterion is the residual translation

        Translation = np.zeros((len(self.ImageOrientationPatient),2))  # 2D translation
        iteration_num = 1

        while stoping_criterion > 3.0 and iteration_num < 100:
            stoping_criterion = 0 
            for i in range(len(self.ImagePositionPatient)):
                # If it is a slice containing the LV
                #print(self.slice_num[i])  
                if len(self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour") & (self.DataSet.slice_number == self.slice_num[i] ),:])>0 :
                                    
                    if len(self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour") & (self.DataSet.slice_number == self.slice_num[i]),:])>0:
                        LVEpiref = self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour") & (self.DataSet.slice_number == self.slice_num[i]),:]
                        #LVEpiref = np.vstack((LVEpiref,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saepicardialContour") & (self.DataSet.slice_number == self.slice_num[i]),:]))
                        #LVEpiref = np.vstack((LVEpiref,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW") & (self.DataSet.slice_number == self.slice_num[i]),:]))
                        #LVEpiref = np.vstack((LVEpiref,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour") & (self.DataSet.slice_number == self.slice_num[i]),:]))
                       
                    Transformation = self.From2Dto3D(self.ImagePositionPatient[i],self.ImageOrientationPatient[i],i)
                    Transformation = np.linalg.inv(Transformation) 
                    pts_LV = np.ones((len(LVEpiref),4))
                    pts_LV[:,0:3] = LVEpiref
                    Px_LV = np.dot(pts_LV,Transformation.T)
                    P2_LV_reference = Px_LV[:,0:2] / (np.vstack((Px_LV[:,3],Px_LV[:,3]))).T

                    LVFixedEndo = np.empty((0,2), float)

                    for j in range(len(self.ImagePositionPatient)):
                        if j!=i:
                            if len(self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour") & (self.DataSet.slice_number == self.slice_num[j]),:])>0:
                                
                                if len(self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour") & (self.DataSet.slice_number == self.slice_num[j]),:])>0:
                                    LVEpi = self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour" ) & (self.DataSet.slice_number == self.slice_num[j]),:]
                                    #LVEpi = np.vstack((LVEpi,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saepicardialContour") & (self.DataSet.slice_number == self.slice_num[j]),:]))
                                    #LVEpi = np.vstack((LVEpi,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW") & (self.DataSet.slice_number == self.slice_num[j]),:]))
                                    #LVEpi = np.vstack((LVEpi,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour") & (self.DataSet.slice_number == self.slice_num[j]),:]))
                             
                                # Compute intersection with Dicom associated with slice i
                                for o in range(len(LVEpi)-1):
                                    P = LineIntersection(self.ImageOrientationPatient[i],self.ImagePositionPatient[i],LVEpi[o,:],LVEpi[o+1,:])

                                    if len(P) > 0 and np.dot(P.T - LVEpi[o,:], P.T -LVEpi[o+1,:]) < 0:
                                        TransformationLVEndo = self.From2Dto3D(self.ImagePositionPatient[i],self.ImageOrientationPatient[i],i)
                                        TransformationLVEndo = np.linalg.inv(TransformationLVEndo) 

                                        pts_LV = np.ones((len(P),4))
                                        pts_LV[:,0:3] = P
                                        Px_LV = np.dot(pts_LV,TransformationLVEndo.T)
                                        P2_LV = Px_LV[:,0:2] / (np.vstack((Px_LV[:,3],Px_LV[:,3]))).T
                                        LVFixedEndo = np.vstack((LVFixedEndo,P2_LV))
                    t=0 # initial value added by Lee
                    if len(LVFixedEndo) > 0:
                        # Find closest point
                        tree = scipy.spatial.cKDTree(P2_LV_reference)
                        d, indx = tree.query(LVFixedEndo, k=1, p=2)

                        funct = lambda x: sum((P2_LV_reference[indx,0]+x[0]-LVFixedEndo[:,0])*(P2_LV_reference[indx,0]+x[0]-LVFixedEndo[:,0]) + (P2_LV_reference[indx,1]+x[1]-LVFixedEndo[:,1])*(P2_LV_reference[indx,1]+x[1]-LVFixedEndo[:,1]))   # (P[0] - P[1])**2 is slower than (P[0] - P[1])*(P[0] - P[1])
                        t = scipy.optimize.fmin(func=funct, x0=[0,0], disp=False)   # Multidimensional unconstrained nonlinear minimization (Nelder-Mead). Starts at X0 = [0,0] and attempts to find a local minimizer 
                    
                    stoping_criterion =  stoping_criterion + np.linalg.norm(t)

                    Translation[i,:] = Translation[i,:] + t

                    # the transation is done in 2D
                    point_2_translate = self.DataSet.Evenly_spaced_points[self.DataSet.slice_number == self.slice_num[i],:]
                    Transformation = self.From2Dto3D(self.ImagePositionPatient[i],self.ImageOrientationPatient[i],i)

                    Points2D = np.ones((len(point_2_translate),4))
                    Points2D[:,0:3] = point_2_translate
                    Px2D_LV = np.dot(Points2D,np.linalg.inv(Transformation) .T)
                    P2D_LV = Px2D_LV[:,0:2] / (np.vstack((Px2D_LV[:,3],Px2D_LV[:,3]))).T

                    LV = P2D_LV + t

                    # Back to 3D
                    pts_LV = np.ones((len(LV),4))
                    pts_LV[:,0:2] = LV
                    pts_LV[:,2] = [0]*len(LV)
                    pts_LV[:,3] = [1]*len(LV)

                    Px_LV = np.dot(Transformation,pts_LV.T)
                    P3_LV = Px_LV[0:3,:] / (np.vstack((Px_LV[3,:],np.vstack((Px_LV[3,:],Px_LV[3,:])))))

                    indexes = np.where((self.DataSet.slice_number == self.slice_num[i]))

                    self.DataSet.Evenly_spaced_points[indexes,:] = P3_LV.T

                # If it is a slice containing the RV only
                if len(self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour") & (self.DataSet.slice_number == self.slice_num[i]),:])==0 and len(self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW") & (self.DataSet.slice_number == self.slice_num[i]),:])>0:
                    #print(self.slice_num[i])                  
                    if len(self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW") & (self.DataSet.slice_number == self.slice_num[i]),:])>0:
                        LVEpiref = self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW") & (self.DataSet.slice_number == self.slice_num[i]),:]
                        LVEpiref = np.vstack((LVEpiref,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVS") & (self.DataSet.slice_number == self.slice_num[i]),:]))
                        #LVEpiref = np.vstack((LVEpiref,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saepicardialContour") & (self.DataSet.slice_number == self.slice_num[i]),:]))
                    
                    Transformation = self.From2Dto3D(self.ImagePositionPatient[i],self.ImageOrientationPatient[i],i)
                    Transformation = np.linalg.inv(Transformation) 
                    pts_LV = np.ones((len(LVEpiref),4))
                    pts_LV[:,0:3] = LVEpiref
                    Px_LV = np.dot(pts_LV,Transformation.T)
                    P2_LV_reference = Px_LV[:,0:2] / (np.vstack((Px_LV[:,3],Px_LV[:,3]))).T

                    LVFixedEndo = np.empty((0,2), float)

                    for j in range(len(self.ImagePositionPatient)):
                        if j!=i:
                            if len(self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW") & (self.DataSet.slice_number == self.slice_num[j]),:])>0:
                                
                                if len(self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW") & (self.DataSet.slice_number == self.slice_num[j]),:])>0:
                                    LVEpi = self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW" ) & (self.DataSet.slice_number == self.slice_num[j]),:]
                                    LVEpi = np.vstack((LVEpi,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVS") & (self.DataSet.slice_number == self.slice_num[j]),:]))
                                    #LVEpi = np.vstack((LVEpi,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saepicardialContour") & (self.DataSet.slice_number == self.slice_num[j]),:]))

                                # Compute intersection with Dicom associated with slice i
                                for o in range(len(LVEpi)-1):
                                    P = LineIntersection(self.ImageOrientationPatient[i],self.ImagePositionPatient[i],LVEpi[o,:],LVEpi[o+1,:])

                                    if len(P) > 0 and np.dot(P.T - LVEpi[o,:], P.T -LVEpi[o+1,:]) < 0:
                                        TransformationLVEndo = self.From2Dto3D(self.ImagePositionPatient[i],self.ImageOrientationPatient[i],i)
                                        TransformationLVEndo = np.linalg.inv(TransformationLVEndo) 

                                        pts_LV = np.ones((len(P),4))
                                        pts_LV[:,0:3] = P
                                        Px_LV = np.dot(pts_LV,TransformationLVEndo.T)
                                        P2_LV = Px_LV[:,0:2] / (np.vstack((Px_LV[:,3],Px_LV[:,3]))).T
                                        LVFixedEndo = np.vstack((LVFixedEndo,P2_LV))
                    t=0 # initial value added by Lee
                    if len(LVFixedEndo) > 0:
                        # Find closest point
                        tree = scipy.spatial.cKDTree(P2_LV_reference)
                        d, indx = tree.query(LVFixedEndo, k=1, p=2)

                        funct = lambda x: sum((P2_LV_reference[indx,0]+x[0]-LVFixedEndo[:,0])*(P2_LV_reference[indx,0]+x[0]-LVFixedEndo[:,0]) + (P2_LV_reference[indx,1]+x[1]-LVFixedEndo[:,1])*(P2_LV_reference[indx,1]+x[1]-LVFixedEndo[:,1]))   # (P[0] - P[1])**2 is slower than (P[0] - P[1])*(P[0] - P[1])
                        t = scipy.optimize.fmin(func=funct, x0=[0,0], disp=False)   # Multidimensional unconstrained nonlinear minimization (Nelder-Mead). Starts at X0 = [0,0] and attempts to find a local minimizer 
                    
                    stoping_criterion =  stoping_criterion + np.linalg.norm(t)

                    Translation[i,:] = Translation[i,:] + t

                    # the transation is done in 2D
                    point_2_translate = self.DataSet.Evenly_spaced_points[self.DataSet.slice_number == self.slice_num[i],:]
                    Transformation = self.From2Dto3D(self.ImagePositionPatient[i],self.ImageOrientationPatient[i],i)

                    Points2D = np.ones((len(point_2_translate),4))
                    Points2D[:,0:3] = point_2_translate
                    Px2D_LV = np.dot(Points2D,np.linalg.inv(Transformation) .T)
                    P2D_LV = Px2D_LV[:,0:2] / (np.vstack((Px2D_LV[:,3],Px2D_LV[:,3]))).T

                    LV = P2D_LV + t

                    # Back to 3D
                    pts_LV = np.ones((len(LV),4))
                    pts_LV[:,0:2] = LV
                    pts_LV[:,2] = [0]*len(LV)
                    pts_LV[:,3] = [1]*len(LV)

                    Px_LV = np.dot(Transformation,pts_LV.T)
                    P3_LV = Px_LV[0:3,:] / (np.vstack((Px_LV[3,:],np.vstack((Px_LV[3,:],Px_LV[3,:])))))

                    indexes = np.where((self.DataSet.slice_number == self.slice_num[i]))

                    self.DataSet.Evenly_spaced_points[indexes,:] = P3_LV.T


            iteration_num = iteration_num+1

        return Translation

    def CreateNextModel(self,DataSetES,ESTranslation):
        """Copy of the current model onto the next time model. Just the dataset is changed.
            Input: 
                DataSetES: dataset for the new time frame
                ESTranslation: 2D translations needed
            Output:
                ESSurface: Copy of the current model ('self'), associated with the new DataSet DataSet.
        """
        
        ESSurface = copy.deepcopy(self)
        ESSurface.DataSet = copy.deepcopy(DataSetES)
        ESSurface.SliceShiftES(ESTranslation,self.ImagePositionPatient)
              
        return ESSurface 

    def SliceShiftES(self,ESTranslation,ImagePositionPatientED):
        """ This function applies 2D translations from breath-hold misregistration correction to the DtaSet self.DataSet.
        
            Input:
                ESTranslation: translations needed (output from LVLAXSAXSliceShifting)        
                ImagePositionPatientED: PatientPosition (needed to jump from Patient coordinates to image coordinates and to identify the corresponding slice)
            Output:
                None. The Dataset 'self.DataSet' is translated in the function itself.          
        """

        for i in range(len(self.DataSet.ImagePositionPatient)):

            if len(self.DataSet.Evenly_spaced_points[(self.DataSet.slice_number == i),:]) >0:
                PtsSlice = self.DataSet.Evenly_spaced_points[(self.DataSet.slice_number == i),:] 

                # Get 2D points  
                # Find the scorrespondign translation by searching the line in ImagePositionPatientED matching with self.DataSet.ImagePositionPatient[i,:]
                index = np.where((self.DataSet.ImagePositionPatient[i] == ImagePositionPatientED).all(axis=1))
                Transformation = self.From2Dto3D(self.DataSet.ImagePositionPatient[i],self.DataSet.ImageOrientationPatient[i],i)
                
                pts_LV = np.ones((len(PtsSlice),4))
                pts_LV[:,0:3] = PtsSlice 
                Px_LV = np.dot(pts_LV,np.linalg.inv(Transformation).T)
                P2 = Px_LV[:,0:2] / (np.vstack((Px_LV[:,3],Px_LV[:,3]))).T
                
                if len(self.DataSet.Evenly_spaced_points[(self.DataSet.slice_number == i) ,:]) > 6: 
                    
                    for j in range(len(P2)):
                        P2[j,0] = P2[j,0] + ESTranslation[index,0]
                        P2[j,1] = P2[j,1] + ESTranslation[index,1]             

                    # Back to 3D   
                    pts = np.ones((len(P2),4))
                    pts[:,0:2] = P2
                    pts[:,2] = [0]*len(P2)
                    pts[:,3] = [1]*len(P2)
                    Px = np.dot(Transformation,pts.T)
                    P3_LV = Px[0:3,:] / (np.vstack((Px[3,:],np.vstack((Px[3,:],Px[3,:])))))                


                    indexes = np.where(self.DataSet.slice_number == i) 
                    contType = self.DataSet.ContourType[indexes]

                    self.DataSet.Evenly_spaced_points = np.delete(self.DataSet.Evenly_spaced_points, indexes, 0)               
                    self.DataSet.Evenly_spaced_points = np.vstack((self.DataSet.Evenly_spaced_points,P3_LV.T))

                    self.DataSet.ContourType = np.delete(self.DataSet.ContourType, indexes, 0)       
                    self.DataSet.slice_number = np.delete(self.DataSet.slice_number, indexes, 0)  

                    self.DataSet.ContourType = np.hstack((self.DataSet.ContourType,contType))
                    self.DataSet.slice_number = np.hstack((self.DataSet.slice_number,[i]*np.asarray(indexes).shape[1])) 
    
    '''def Generate_New_Scale_Sparse(self): 
        """ This function transforms matrices mBder into a sparse matrix. This is the format supported by CPLEX 
            Input:
                None
            Output:
                final_rows: mBder matrices in the format supported by CPLEX
            """  
     
        final_rows = []
             
        for i in range(len(self.mBder_dx)):  # rows and colums will always be the same so we just need to precompute this and then calculate the values...
     
            dXdxi = np.zeros((3,3),dtype = 'float') 
                             
            dXdxi[:,0] = np.dot(self.mBder_dx[i,:],self.control_mesh)
            dXdxi[:,1] = np.dot(self.mBder_dy[i,:],self.control_mesh)
            dXdxi[:,2] = np.dot(self.mBder_dz[i,:],self.control_mesh)
                 
            g = np.linalg.inv(dXdxi)
                
            Gx = np.dot(self.mBder_dx[i,:],g[0,0]) + np.dot(self.mBder_dy[i,:],g[1,0]) + np.dot(self.mBder_dz[i,:],g[2,0])         
            indices_column = np.nonzero(Gx)[0]   # get row index
                
            cols = []
            vals = []          
            for j in range(len(indices_column)):
                cols.append(int(indices_column[j]))
                vals.append(Gx[indices_column[j]])
     
            final_rows.append(cplex.SparsePair(ind = list(cols), val = list(vals)))
                         
            Gy = np.dot(self.mBder_dx[i,:],g[0,1]) + np.dot(self.mBder_dy[i,:],g[1,1]) + np.dot(self.mBder_dz[i,:],g[2,1])
            indices_column = np.nonzero(Gy)[0]   # get row index 
            cols = []
            vals = []           
            for j in range(len(indices_column)):
                cols.append(int(indices_column[j]))
                vals.append(Gy[indices_column[j]])
                                 
            final_rows.append(cplex.SparsePair(ind = list(cols), val = list(vals)))
                
            Gz = np.dot(self.mBder_dx[i,:],g[0,2]) + np.dot(self.mBder_dy[i,:],g[1,2]) + np.dot(self.mBder_dz[i,:],g[2,2])
            indices_column = np.nonzero(Gz)[0]   # get row index 
            cols = []
            vals = []           
            for j in range(len(indices_column)):
                cols.append(int(indices_column[j]))
                vals.append(Gz[indices_column[j]])
    
            final_rows.append(cplex.SparsePair(ind = list(cols), val = list(vals)))
            
        return final_rows'''

    def UpdatePoseAndScale(self):
        """ A method that initializes the model. It takes a DataSet and updates the model pose and scale
            in accordance with the data points. 

            Input:
                None
            Output:
                scaleFactor: scale factor between template and data points.
        """ 

        base = self.DataSet.Mitral_centroid

        # Get centroid RV        
        RVS_points_Model = self.etPos[self.etVertexStartEnd[1,0]:self.etVertexStartEnd[2,1],:]
 
        RV_GP = self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW"),0:3] 
        RV_GP = np.vstack((RV_GP,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVS"),0:3] ))
        rv = RV_GP.mean(axis=0)
        rv_model = RVS_points_Model.mean(axis=0)

        xaxis = self.DataSet.Apex - base
        xaxis = xaxis / np.linalg.norm(xaxis)
        
        xaxis_model = self.etPos[self.Apex_index,:] - self.etPos[self.etVertexStartEnd[4,1],:]
        xaxis_model = xaxis_model / np.linalg.norm(xaxis_model)

        tempOrigin = 0.5*(self.DataSet.Apex + base)
        tempOrigin_model = self.etPos[self.Apex_index,:]

        maxd = np.linalg.norm(0.5*(self.DataSet.Apex - base))
        mind = -np.linalg.norm(0.5*(self.DataSet.Apex - base))

        maxd_model = np.linalg.norm(0.5*(self.etPos[self.Apex_index,:] - self.etPos[self.etVertexStartEnd[4,1],:]))
        mind_model = -np.linalg.norm(0.5*(self.etPos[self.Apex_index,:] - self.etPos[self.etVertexStartEnd[4,1],:]))
           
        point_proj = self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW"),:]
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "larvendocardialContour"),:]))        
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "sarvendocardialContour"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVS"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "laendocardialContour"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "BP_point"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "LA_Apex_Point"),:]))        
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "Tricuspid_Valve"),:]))      
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "Pulmonary"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "Aorta"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RV_insert"),:]))

        for i in range(len(point_proj)):
            tempd = np.dot(xaxis,(point_proj[i,:]-tempOrigin))
            if tempd > maxd:
                maxd = tempd 
                    
            if tempd < mind:    
                mind = tempd

        scaleFactorModel = np.linalg.norm(self.etPos[self.Apex_index,:]- self.etPos[self.etVertexStartEnd[4,1],:])  
        scaleFactor = 0.9*np.linalg.norm(self.DataSet.Apex-base)/scaleFactorModel

        point_used_model = self.etPos
        for i in range(len(point_used_model)):
            tempd_model = np.dot(xaxis_model,(point_used_model[i,:]-tempOrigin_model))
            if tempd_model > maxd_model:
                maxd_model = tempd_model 
                    
            if tempd_model < mind_model:    
                mind_model = tempd_model
        
        self.control_mesh = self.control_mesh*scaleFactor
        self.etPos = np.dot(self.matrix,self.control_mesh)
        
        centroid = tempOrigin + mind*xaxis + ((maxd-mind)/3.0)*xaxis
        centroid_model = tempOrigin_model + mind_model*xaxis_model + ((maxd_model-mind_model)/3.0)*xaxis_model

        scale = np.dot(xaxis, rv) - np.dot(xaxis, centroid)/np.dot(xaxis,xaxis)
        scale_model = np.dot(xaxis_model, rv_model) - np.dot(xaxis_model, centroid_model)/np.dot(xaxis_model,xaxis_model)
        
        rvproj = centroid + scale*xaxis
        rvproj_model = centroid_model + scale_model*xaxis_model
        
        yaxis = rv - rvproj
        yaxis_model = rv_model - rvproj_model
        
        yaxis = yaxis / np.linalg.norm(yaxis)
        yaxis_model = yaxis_model / np.linalg.norm(yaxis_model)
        
        zaxis = np.cross(xaxis,yaxis)
        zaxis_model = np.cross(xaxis_model,yaxis_model)
        
        zaxis = zaxis / np.linalg.norm(zaxis)
        zaxis_model = zaxis_model / np.linalg.norm(zaxis_model)
      
        # Find translation and rotation between the two frames of reference
        """ The easiest way to solve it (in my opinion) is by using a Singular Value Decomposition as reported by Markley (1988):
            1. Obtain a matrix B as follows:
                B=∑ni=1aiwiviTB=∑i=wiviT
            2. Find the SVD of BB
                B=USVT
            3. The rotation matrix is:
                R=UMVT, where M=diag([11det(U)det(V)])
        """

        # The rotation is defined about the origin so we need to translate the model to the origin 
        for t in range(self.numNodes):
            self.control_mesh[t,:] = np.add(self.control_mesh[t,:],-self.etPos.mean(axis=0))
            
        self.etPos = np.dot(self.matrix,self.control_mesh)  # update etPos

        # Step 1
        B = np.zeros((3,3))
        B = np.outer(xaxis,xaxis_model) + np.outer(yaxis,yaxis_model) + np.outer(zaxis,zaxis_model)
        
        # Step 2
        [U, s, Vt] = np.linalg.svd(B) 
           
        M = np.array([[1,0,0],[0,1,0],[0,0,np.linalg.det(U) * np.linalg.det(Vt)]])
        Rotation = np.dot(U,np.dot(M,Vt))
  
        for t in range(self.numNodes):
            self.control_mesh[t,:] = np.dot(Rotation,self.control_mesh[t,:]) 
                
        self.etPos = np.dot(self.matrix,self.control_mesh)

        # Translate the model back to origin of the DataSet coordinate system
        translation_to_apply = 0.5*((self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW"),:]).mean(axis=0) + (self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour"),:]).mean(axis=0))
  
        for t in range(self.numNodes):
            self.control_mesh[t,:] = np.add(self.control_mesh[t,:],translation_to_apply)

        self.etPos = np.dot(self.matrix,self.control_mesh) # etPos update

        return scaleFactor

    def UpdatePoseAndScale(self):
        """ A method that initializes the model. It takes a DataSet and updates the model pose and scale
            in accordance with the data points. 

            Input:
                None
            Output:
                scaleFactor: scale factor between template and data points.
        """ 

        base = self.DataSet.Mitral_centroid

        # Get centroid RV        
        RVS_points_Model = self.etPos[self.etVertexStartEnd[1,0]:self.etVertexStartEnd[2,1],:]
 
        RV_GP = self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW"),0:3] 
        RV_GP = np.vstack((RV_GP,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVS"),0:3] ))
        rv = RV_GP.mean(axis=0)
        rv_model = RVS_points_Model.mean(axis=0)

        xaxis = self.DataSet.Apex - base
        xaxis = xaxis / np.linalg.norm(xaxis)
        
        xaxis_model = self.etPos[self.Apex_index,:] - self.etPos[self.etVertexStartEnd[4,1],:]
        xaxis_model = xaxis_model / np.linalg.norm(xaxis_model)

        tempOrigin = 0.5*(self.DataSet.Apex + base)
        tempOrigin_model = self.etPos[self.Apex_index,:]

        maxd = np.linalg.norm(0.5*(self.DataSet.Apex - base))
        mind = -np.linalg.norm(0.5*(self.DataSet.Apex - base))

        maxd_model = np.linalg.norm(0.5*(self.etPos[self.Apex_index,:] - self.etPos[self.etVertexStartEnd[4,1],:]))
        mind_model = -np.linalg.norm(0.5*(self.etPos[self.Apex_index,:] - self.etPos[self.etVertexStartEnd[4,1],:]))
           
        point_proj = self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW"),:]
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saepicardialContour"),:]))        
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVS"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "BP_point"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "LA_Apex_Point"),:]))        
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "Tricuspid_Valve"),:]))      
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "Pulmonary"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "Aorta"),:]))
        point_proj = np.vstack((point_proj,self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RV_insert"),:]))

        for i in range(len(point_proj)):
            tempd = np.dot(xaxis,(point_proj[i,:]-tempOrigin))
            if tempd > maxd:
                maxd = tempd 
                    
            if tempd < mind:    
                mind = tempd

        scaleFactorModel = np.linalg.norm(self.etPos[self.Apex_index,:]- self.etPos[self.etVertexStartEnd[4,1],:])  
        scaleFactor = 0.9*np.linalg.norm(self.DataSet.Apex-base)/scaleFactorModel

        point_used_model = self.etPos
        for i in range(len(point_used_model)):
            tempd_model = np.dot(xaxis_model,(point_used_model[i,:]-tempOrigin_model))
            if tempd_model > maxd_model:
                maxd_model = tempd_model 
                    
            if tempd_model < mind_model:    
                mind_model = tempd_model
        
        self.control_mesh = self.control_mesh*scaleFactor
        self.etPos = np.dot(self.matrix,self.control_mesh)
        
        centroid = tempOrigin + mind*xaxis + ((maxd-mind)/3.0)*xaxis
        centroid_model = tempOrigin_model + mind_model*xaxis_model + ((maxd_model-mind_model)/3.0)*xaxis_model

        scale = np.dot(xaxis, rv) - np.dot(xaxis, centroid)/np.dot(xaxis,xaxis)
        scale_model = np.dot(xaxis_model, rv_model) - np.dot(xaxis_model, centroid_model)/np.dot(xaxis_model,xaxis_model)
        
        rvproj = centroid + scale*xaxis
        rvproj_model = centroid_model + scale_model*xaxis_model
        
        yaxis = rv - rvproj
        yaxis_model = rv_model - rvproj_model
        
        yaxis = yaxis / np.linalg.norm(yaxis)
        yaxis_model = yaxis_model / np.linalg.norm(yaxis_model)
        
        zaxis = np.cross(xaxis,yaxis)
        zaxis_model = np.cross(xaxis_model,yaxis_model)
        
        zaxis = zaxis / np.linalg.norm(zaxis)
        zaxis_model = zaxis_model / np.linalg.norm(zaxis_model)
      
        # Find translation and rotation between the two frames of reference
        """ The easiest way to solve it (in my opinion) is by using a Singular Value Decomposition as reported by Markley (1988):
            1. Obtain a matrix B as follows:
                B=∑ni=1aiwiviTB=∑i=wiviT
            2. Find the SVD of BB
                B=USVT
            3. The rotation matrix is:
                R=UMVT, where M=diag([11det(U)det(V)])
        """

        # The rotation is defined about the origin so we need to translate the model to the origin 
        for t in range(self.numNodes):
            self.control_mesh[t,:] = np.add(self.control_mesh[t,:],-self.etPos.mean(axis=0))
            
        self.etPos = np.dot(self.matrix,self.control_mesh)  # update etPos

        # Step 1
        B = np.zeros((3,3))
        B = np.outer(xaxis,xaxis_model) + np.outer(yaxis,yaxis_model) + np.outer(zaxis,zaxis_model)
        
        # Step 2
        [U, s, Vt] = np.linalg.svd(B) 
           
        M = np.array([[1,0,0],[0,1,0],[0,0,np.linalg.det(U) * np.linalg.det(Vt)]])
        Rotation = np.dot(U,np.dot(M,Vt))
  
        for t in range(self.numNodes):
            self.control_mesh[t,:] = np.dot(Rotation,self.control_mesh[t,:]) 
                
        self.etPos = np.dot(self.matrix,self.control_mesh)

        # Translate the model back to origin of the DataSet coordinate system
        translation_to_apply = 0.5*((self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "RVFW"),:]).mean(axis=0) + (self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour"),:]).mean(axis=0))
  
        for t in range(self.numNodes):
            self.control_mesh[t,:] = np.add(self.control_mesh[t,:],translation_to_apply)

        self.etPos = np.dot(self.matrix,self.control_mesh) # etPos update

        return scaleFactor

    def Generate_Contraint_Matrix(self): 
        """ This function generates constraints matrix to be given to cvxopt 
            Input:
                None
            Output:
                constraints: constraints matrix
            """  

        constraints = []
        for i in range(len(self.mBder_dx)):  # rows and colums will always be the same so we just need to precompute this and then calculate the values...

            dXdxi = np.zeros((3,3),dtype = 'float') 
                        
            dXdxi[:,0] = np.dot(self.mBder_dx[i,:],self.control_mesh)
            dXdxi[:,1] = np.dot(self.mBder_dy[i,:],self.control_mesh)
            dXdxi[:,2] = np.dot(self.mBder_dz[i,:],self.control_mesh)
            
            g = np.linalg.inv(dXdxi)
           
            Gx = np.dot(self.mBder_dx[i,:],g[0,0]) + np.dot(self.mBder_dy[i,:],g[1,0]) + np.dot(self.mBder_dz[i,:],g[2,0])         
            constraints.append(Gx)
                    
            Gy = np.dot(self.mBder_dx[i,:],g[0,1]) + np.dot(self.mBder_dy[i,:],g[1,1]) + np.dot(self.mBder_dz[i,:],g[2,1])      
            constraints.append(Gy)
            
            Gz = np.dot(self.mBder_dx[i,:],g[0,2]) + np.dot(self.mBder_dy[i,:],g[1,2]) + np.dot(self.mBder_dz[i,:],g[2,2])
            constraints.append(Gz)
            
        return np.asmatrix(constraints)  

    '''def SolveProblem(self, linear_part_x,linear_part_y,linear_part_z,constant_part_x,constant_part_y,constant_part_z,quadratic_part):
        """ This function solves the quadratic constrained problem. This problem has the following form:
            ||AX-d||^2 = X'*A'*A*X -2*d'*A*X + d'*d. x, y and z directions are solved separately.
            Tips for Programming Large Models: Batching preferred, Manage variables/constraints by indices, Program in Python style, Python has a built-in profiler 
            
            Input:
                linear_part_x,linear_part_y,linear_part_z: linear parts (-2*dt*A*x)
                constant_part_x,constant_part_y,constant_part_z: constant parts (dt*d)
                quadratic_part: quadratic part (xt*At*A*x)

            Output:
                solx,soly,solz: solutions of the problem
        """
           
        cx = cplex.Cplex()        
        cy = cplex.Cplex()
        cz = cplex.Cplex()
        
        cx.objective.set_sense(cx.objective.sense.minimize)
        cy.objective.set_sense(cy.objective.sense.minimize)
        cz.objective.set_sense(cz.objective.sense.minimize)
      
        # Print out log file if needed
        #out = cx.set_results_stream("C:/Users/cmau619/Desktop/Toy_problem_python/logx.txt")       
        #out = cy.set_results_stream("C:/Users/cmau619/Desktop/Toy_problem_python/logy.txt")        
        #out = cz.set_results_stream("C:/Users/cmau619/Desktop/Toy_problem_python/logz.txt")
        cx.set_results_stream(None)
        cy.set_results_stream(None) 
        cz.set_results_stream(None) 
#        
        my_lb = -200*np.ones(self.numNodes)
        my_ub = 200*np.ones(self.numNodes) 
#      
        my_obj_x = linear_part_x        
        my_obj_y = linear_part_y 
        my_obj_z = linear_part_z 
                            
        cx.variables.add(obj = my_obj_x,lb = my_lb,ub = my_ub) # Sets the linear part of the objective function -dt*A*x  
        cy.variables.add(obj = my_obj_y,lb = my_lb,ub = my_ub)  
        cz.variables.add(obj = my_obj_z,lb = my_lb,ub = my_ub) 

        cx.objective.set_quadratic(quadratic_part)   # needs to be placed after c.variables.add(obj = my_obj... the quadratic part is the same for x,y and z
        cy.objective.set_quadratic(quadratic_part)   # needs to be placed after c.variables.add(obj = my_obj...
        cz.objective.set_quadratic(quadratic_part)   # needs to be placed after c.variables.add(obj = my_obj... 
#                
        cx.objective.set_offset(constant_part_x)          
        cy.objective.set_offset(constant_part_y)  
        cz.objective.set_offset(constant_part_z) 
#        
        rows = self.Generate_New_Scale_Sparse()

        bound = (1/3)
        size = 3*len(self.mBder_dx) 
        cx.linear_constraints.add(lin_expr = rows,
                                  senses = ["L"]*size,
                                  rhs = [bound]*size)
                                                                      
        cx.linear_constraints.add(lin_expr = rows,
                                  senses = ["G"]*size,
                                  rhs = [-bound]*size)
                                                            
        cy.linear_constraints.add(lin_expr = rows,
                                  senses = ["L"]*size,
                                  rhs = [bound]*size) 

        cy.linear_constraints.add(lin_expr = rows,
                                  senses = ["G"]*size,
                                  rhs = [-bound]*size)
                                   
        cz.linear_constraints.add(lin_expr = rows,
                                  senses = ["L"]*size,
                                  rhs = [bound]*size) 

        cz.linear_constraints.add(lin_expr = rows,
                                  senses = ["G"]*size,
                                  rhs = [-bound]*size)     
        
        try:                       
            cx.solve()
        except cplex.CplexSolverError:
            print("Exception raised during solve x")
            return
        solx = cx.solution.get_values()           
            
        try:        
            cy.solve()
        except cplex.CplexSolverError:
            print("Exception raised during solve y")
            return 
        soly = cy.solution.get_values()
          
        try:         
            cz.solve()
        except cplex.CplexSolverError:
            print("Exception raised during solve z")
            return 
        solz = cz.solution.get_values() 
       
        
        return [solx,soly,solz]'''
       
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
        
        slice_num = []
        for item in SliceID:
            for a in item:
                slice_num.append(int(a))

        return [ImagePositionPatient, ImageOrientationPatient, slice_num, PixelSpacing]
       
  
    def MultiThreadSmoothingED(self,case,weight_GP):
        """ This function performs a series of LLS fits. At each iteration the least squares optimisation is performed and the determinant of the 
            Jacobian matrix is calculated. If all the values are positive, the subdivision surface is deformed by updating its control points, projections 
            are recalculated and the regularization weight is decreased. As long as the deformation is diffeomorphic, smoothing weight is decreased.
            Input:
                case: case name
                weight_GP: data_points' weight
            Output:
                None. 'self' is updated in the function itself
        """
        start_time = time.time()
        High_weight = 10E+10      # First regularization weight
        Isdiffeo = 1
        iteration = 1     
        factor = 5                              
        IsfirstTimeNonDiffeo = 3  
        min_jacobian = 0.1      

        while (Isdiffeo == 1) & (High_weight > 100) :
            print('Iteration #'+str(iteration) +' for the implicitly constrained fit')
            [full_matrix_coefficient_points,index,d,w_out,dist_prior] = self.CalcDataSetXi(weight_GP)

            PriorPosition = np.linalg.multi_dot([full_matrix_coefficient_points,self.control_mesh])

            w = w_out*np.identity(len(PriorPosition))
            WPG = np.linalg.multi_dot([w,full_matrix_coefficient_points])                                                                
            GTPTWTWPG = np.linalg.multi_dot([WPG.T,WPG])  # np.linalg.multi_dot faster than np.dot
            A = GTPTWTWPG + High_weight*(self.GTSTSG_x + self.GTSTSG_y + 0.001*self.GTSTSG_z)
            Wd = np.linalg.multi_dot([w,d-PriorPosition])
            rhs = np.linalg.multi_dot([WPG.T,Wd])  

            solf = np.linalg.solve(A.T.dot(A),A.T.dot(rhs))  # solve the Moore-Penrose pseudo inversee
            Isdiffeo = self.IsDiffeomorphic(np.add(self.control_mesh,solf),min_jacobian)

            if Isdiffeo == 1: 
                self.control_mesh = np.add(self.control_mesh,solf)
                self.etPos = np.linalg.multi_dot([self.matrix,self.control_mesh])
                High_weight = High_weight/factor   # we divide weight by 'factor' and start again...
                PriorPosition = np.linalg.multi_dot([full_matrix_coefficient_points,self.control_mesh])
                err = np.mean(np.sqrt( (d[:,0]-PriorPosition[:,0])**2 + (d[:,2]-PriorPosition[:,2])**2 + (d[:,1]-PriorPosition[:,1])**2))

            else: 
                # If Isdiffeo ==1, the model is not updated. We divide factor by 2 and try again. After the third time (IsfirstTimeNonDiffeo = 3 at the beginning), we stop.  
                if factor > 1:
                    IsfirstTimeNonDiffeo = IsfirstTimeNonDiffeo-1
                    factor = factor/2
                    High_weight = High_weight*factor
                    Isdiffeo = 1 

            iteration = iteration+1 

        print("End of the implicitly constrained fit")
        print("--- %s seconds ---" % (time.time() - start_time))  

    '''def MultiThreadSmoothingDiffeoED(self,case,weight_GP,low_smoothing_weight):
        """ This function performs the proper diffeomorphic fit.

            Input:
                case: case name 

            Output:
                None. 'self' is updated in the function itself
        """
        start_time = time.time()
        [full_matrix_coefficient_points, index, d, w_out, dist_prior] = self.CalcDataSetXi(weight_GP)

        # Remove outliers
        full_matrix_coefficient_points=full_matrix_coefficient_points[abs(dist_prior-np.mean(dist_prior)) < 6 * np.std(dist_prior) ,:]
        index = index[abs(dist_prior-np.mean(dist_prior)) < 6 * np.std(dist_prior)]
        d = d[abs(dist_prior-np.mean(dist_prior)) < 6 * np.std(dist_prior),:]
        w_out = w_out[abs(dist_prior-np.mean(dist_prior)) < 6 * np.std(dist_prior)]
        #contType = contType[abs(dist_prior-np.mean(dist_prior)) < 6 * np.std(dist_prior)]       
        
        PriorPosition = np.dot(full_matrix_coefficient_points,self.control_mesh)
 
        w = np.eye(len(PriorPosition))
        for i in range(len(PriorPosition)):
            w[i,i] = w_out[i]

        WPG = np.dot(w,full_matrix_coefficient_points)                                                                
        GTPTWTWPG = np.dot(WPG.T,WPG)
#
        A = GTPTWTWPG + low_smoothing_weight*(self.GTSTSG_x + self.GTSTSG_y + 1000*self.GTSTSG_z)
        Wd = np.dot(w,d-PriorPosition)
        rhs = np.dot(WPG.T,Wd)

        previous_step_err = 11
        step_err = 10
        tol = 1e-3
        iteration = 0
#
        B = 2*A                                   # set_quadratic multiply by 0.5 because this is the form of a quadratic problem                
        quadratic_form = 0.5*(B+B.T)              # to make it symmetric.
# 
        qmat = []        
        for i in range(self.numNodes):  # can be outside of the function as it does not change...
            qmat.append(cplex.SparsePair(ind = list(range(self.numNodes)), val = list(quadratic_form[i,:])))

        prev_displacement = np.zeros((self.numNodes,3)) 
        

        while abs(step_err-previous_step_err) > tol and iteration < 7:
            print(iteration)
            previous_step_err = np.sqrt(np.mean(  (d[:,0]-PriorPosition[:,0])*(d[:,0]-PriorPosition[:,0]) + (d[:,2]-PriorPosition[:,2])*(d[:,2]-PriorPosition[:,2]) + (d[:,1]-PriorPosition[:,1])*(d[:,1]-PriorPosition[:,1])) )

            linear_part_x = 2*np.dot(prev_displacement[:,0].T,A) - 2*np.dot(Wd[:,0].T,WPG) 
            linear_part_y = 2*np.dot(prev_displacement[:,1].T,A) - 2*np.dot(Wd[:,1].T,WPG)
            linear_part_z = 2*np.dot(prev_displacement[:,2].T,A) - 2*np.dot(Wd[:,2].T,WPG)

            constant_part_x = np.dot(np.dot(prev_displacement[:,0].T,A),prev_displacement[:,0]) + np.dot(Wd[:,0].T,Wd[:,0]) -2* np.dot(np.dot(Wd[:,0].T,WPG),prev_displacement[:,0])                 
            constant_part_y = np.dot(np.dot(prev_displacement[:,1].T,A),prev_displacement[:,1]) + np.dot(Wd[:,1].T,Wd[:,1]) -2* np.dot(np.dot(Wd[:,1].T,WPG),prev_displacement[:,1])
            constant_part_z = np.dot(np.dot(prev_displacement[:,2].T,A),prev_displacement[:,2]) + np.dot(Wd[:,2].T,Wd[:,2]) -2* np.dot(np.dot(Wd[:,2].T,WPG),prev_displacement[:,2])

            [solx, soly, solz] = self.SolveProblem(linear_part_x,linear_part_y,linear_part_z,constant_part_x,constant_part_y,constant_part_z,qmat)

            displacement = np.zeros((self.numNodes,3))
            displacement[:,0] = solx
            displacement[:,1] = soly
            displacement[:,2] = solz

            # check if diffeomorphic 
            Isdiffeo = self.IsDiffeomorphic(np.add(self.control_mesh,displacement),0.1) 

            if Isdiffeo == 0:
                # Due to numerical approximations, epicardium and endocardium can 'touch' (but not cross), leading to a negative jacobian. If it happens, we stop. 
                diffeo = 0
                break

            else:
                prev_displacement[:,0] = prev_displacement[:,0] + solx
                prev_displacement[:,1] = prev_displacement[:,1] + soly
                prev_displacement[:,2] = prev_displacement[:,2] + solz

                self.control_mesh = self.control_mesh + displacement
                self.etPos = np.dot(self.matrix,self.control_mesh)
                PriorPosition = np.dot(full_matrix_coefficient_points,self.control_mesh) 
                
                step_err = np.mean(np.sqrt( (d[:,0]-PriorPosition[:,0])**2 + (d[:,2]-PriorPosition[:,2])**2 + (d[:,1]-PriorPosition[:,1])**2))

                iteration = iteration+1     
        
        err = np.mean(np.sqrt( (d[:,0]-PriorPosition[:,0])**2 + (d[:,2]-PriorPosition[:,2])**2 + (d[:,1]-PriorPosition[:,1])**2))
        print(err)
        print("--- %s seconds CPLEX ---" % (time.time() - start_time))''' 

    def FitModel(self,case,time_frame,saving_path,weight_GP,low_smoothing_weight,transmural_weight):
        """ This function creates mitral, tricuspid and RV epicardial phantom points and calls MultiThreadSmoothingED and MultiThreadSmoothingDiffeoED. 
            Mitral and tricuspid phantom points are created to force the valve to stay circular. RV epicardial phantom points are created to 'help' as the RV epicardium was not contoured 
            in the UK Biobank (if your dataset contains RV epicardial points, you don't need to do it). If you have points on the aorta and pulmonary, you can create 
            phantom points (and I recommend it). 

            Input:
                case: case name
                time_frame: time frame (ED = 1)
                saving_path: path where models and contours are going to be saved.
              
            Output:
                None 
        """ 

        BP_points = self.DataSet.Create_Mitral_phantomPoints(30)
        self.DataSet.Evenly_spaced_points = np.vstack((self.DataSet.Evenly_spaced_points,BP_points))
        self.DataSet.slice_number = np.hstack((self.DataSet.slice_number,[-1]*len(BP_points)))
        self.DataSet.ContourType = np.hstack((self.DataSet.ContourType,["BP_phantom"]*len(BP_points))) 

        Tri_points = self.DataSet.Create_Tricuspid_phantomPoints(30)
        self.DataSet.Evenly_spaced_points = np.vstack((self.DataSet.Evenly_spaced_points,Tri_points))
        self.DataSet.slice_number = np.hstack((self.DataSet.slice_number,[-1]*len(Tri_points)))
        self.DataSet.ContourType = np.hstack((self.DataSet.ContourType,["Tricuspid_phantom"]*len(Tri_points))) 

        RV_epi = self.DataSet.Create_RV_epi()
        self.DataSet.Evenly_spaced_points = np.vstack((self.DataSet.Evenly_spaced_points,RV_epi[:,0:3]))
        self.DataSet.slice_number = np.hstack((self.DataSet.slice_number,RV_epi[:,4]))
        self.DataSet.ContourType = np.hstack((self.DataSet.ContourType,["RVepicardialContour"]*len(RV_epi[:,3])))  

        # Circle does not separate LAX RV contours into RVS and RVFW so we need to do it ourselves. Again, you can comment it if you don't have this issue.
        self.DataSet.Identify_RVS_LAX()

        self.MultiThreadSmoothingED(case,weight_GP)
        self.SolveProblemCVXOPT(case,weight_GP,low_smoothing_weight,transmural_weight)

    def SolveProblemCVXOPT(self,case,weight_GP,low_smoothing_weight,transmural_weight):
        """ This function performs the proper diffeomorphic fit.
            Input:
                case: case name 
                weight_GP: data_points' weight
                low_smoothing_weight: smoothing weight (for regularization term)
            Output:
                None. 'self' is updated in the function itself
        """
        start_time = time.time()
        [full_matrix_coefficient_points, index, d, w_out,dist_prior] = self.CalcDataSetXi(weight_GP) 

        #full_matrix_coefficient_points=full_matrix_coefficient_points[abs(dist_prior-np.mean(dist_prior)) < 6 * np.std(dist_prior) ,:]
        #index = index[abs(dist_prior-np.mean(dist_prior)) < 6 * np.std(dist_prior)]
        #d = d[abs(dist_prior-np.mean(dist_prior)) < 6 * np.std(dist_prior),:]
        #w_out = w_out[abs(dist_prior-np.mean(dist_prior)) < 6 * np.std(dist_prior)]


        PriorPosition = np.dot(full_matrix_coefficient_points,self.control_mesh)
        w = w_out*np.identity(len(PriorPosition))
        WPG = np.dot(w,full_matrix_coefficient_points)                                                                
        GTPTWTWPG = np.dot(WPG.T,WPG)

        A = GTPTWTWPG + low_smoothing_weight*(self.GTSTSG_x + self.GTSTSG_y + transmural_weight*self.GTSTSG_z)
        Wd = np.dot(w,d-PriorPosition)
        rhs = np.dot(WPG.T,Wd)

        previous_step_err = 11
        step_err = 10
        tol = 1e-3
        iteration = 0

        Q = 2*A #.T*A  # 2*A 
        quadratic_form = matrix(0.5*(Q+Q.T), tc='d') # to make it symmetrical.

        prev_displacement = np.zeros((self.numNodes,3)) 

        saved_error = []

        while abs(step_err-previous_step_err) > tol and iteration < 7: 
            print('Iteration #'+str(iteration+1) +' for the explicitly constrained fit')
            previous_step_err = np.sqrt(np.mean((d[:,0]-PriorPosition[:,0])*(d[:,0]-PriorPosition[:,0]) + (d[:,2]-PriorPosition[:,2])*(d[:,2]-PriorPosition[:,2]) + (d[:,1]-PriorPosition[:,1])*(d[:,1]-PriorPosition[:,1])) )

            linear_part_x = matrix((2*np.dot(prev_displacement[:,0].T,A) - 2*np.dot(Wd[:,0].T,WPG).T), tc='d') 
            linear_part_y = matrix((2*np.dot(prev_displacement[:,1].T,A) - 2*np.dot(Wd[:,1].T,WPG).T), tc='d')
            linear_part_z = matrix((2*np.dot(prev_displacement[:,2].T,A) - 2*np.dot(Wd[:,2].T,WPG).T), tc='d')

            linConstraints = matrix(self.Generate_Contraint_Matrix(), tc='d')
            linConstraintNeg = -linConstraints

            G = matrix(np.vstack((linConstraints,linConstraintNeg)))
            size = 2*(3*len(self.mBder_dx))
            bound = 1/3
            h = matrix([bound]*size)

            solvers.options['show_progress'] = False

            #  Solver: solvers.qp(P,q,G,h) see https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf for explanation
            solx=solvers.qp(quadratic_form, linear_part_x, G, h)
            soly=solvers.qp(quadratic_form, linear_part_y, G, h)
            solz=solvers.qp(quadratic_form, linear_part_z, G, h)

            sx = []
            sy = []
            sz = []

            for a in solx['x']:
                sx.append(a)

            for a in soly['x']:
                sy.append(a)

            for a in solz['x']:
                sz.append(a)

            displacement = np.zeros((self.numNodes,3))

            displacement[:,0] = np.asarray(sx)
            displacement[:,1] = np.asarray(sy)
            displacement[:,2] = np.asarray(sz)

            # check if diffeomorphic 
            Isdiffeo = self.IsDiffeomorphic(np.add(self.control_mesh,displacement),0.1) 

            if Isdiffeo == 0:
                # Due to numerical approximations, epicardium and endocardium can 'touch' (but not cross), leading to a negative jacobian. If it happens, we stop.
                diffeo = 0
                break

            else:
                prev_displacement[:,0] = prev_displacement[:,0] + sx
                prev_displacement[:,1] = prev_displacement[:,1] + sy
                prev_displacement[:,2] = prev_displacement[:,2] + sz

                self.control_mesh = self.control_mesh + displacement
                self.etPos = np.dot(self.matrix,self.control_mesh)
                PriorPosition = np.dot(full_matrix_coefficient_points,self.control_mesh) 
                step_err = np.mean(np.sqrt( (d[:,0]-PriorPosition[:,0])**2 + (d[:,2]-PriorPosition[:,2])**2 + (d[:,1]-PriorPosition[:,1])**2))
                iteration = iteration+1
      
        err = np.mean(np.sqrt( (d[:,0]-PriorPosition[:,0])**2 + (d[:,2]-PriorPosition[:,2])**2 + (d[:,1]-PriorPosition[:,1])**2))

        print("--- End of the explicitly constrained fit ---")
        print("--- %s seconds ---" % (time.time() - start_time))
    def FitNextModel(self,filename,filenameInfo,case,saving_path,Translation,time_frame,weight_GP,low_smoothing_weight,transmural_weight):
        """ This function initializes a new model by copying self, changes time_frame data (contour points associated with time_frame) and calls FitModel.

            Input:
                filename: filename is the file containing the 3D contour points coordinates, labels and time frame (see example GPfile.txt).
                filenameInfo: filename is the file containing dicom info  (see example SliceInfoFile.txt).
                case: case number
                Translation: translation needed to correct breath-hold misregistration
                time_frame: time frame #
                saving_path: path where models and contours are going to be saved.
              
            Output:
                None 
        """

        print('Working on time frame #'+str(time_frame) + ' ...')

        surface = copy.deepcopy(self)
        DataSet = GPDataSet(filename,filenameInfo,case,time_frame) 
        surface = surface.CreateNextModel(DataSet,Translation)
        surface.FitModel(case,time_frame,saving_path,weight_GP,low_smoothing_weight,transmural_weight)

        return surface

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
    
    def PlotSurface(self, face_color_LV, face_color_RV,face_color_epi,my_name, surface = "all",opacity = 0.8):
        """ Plot 3D model. 
            Input:
               face_color_LV, face_color_RV, face_color_epi: LV, RV and epi colors
               my_name: surface name
               surface (optional): all = entire surface, endo = endocardium, epi = epicardium  (default = "all")
            Output:
               triangles_epi, triangles_LV, triangles_RV: triangles that need to be plotted for the epicardium, LV and Rv respectively
               lines: lines that need to be plotted
        """ 

        x = np.array(self.etPos[:,0]).T
        y = np.array(self.etPos[:,1]).T
        z = np.array(self.etPos[:,2]).T

        # LV endo
        I_LV = np.asarray(self.ETIndices[self.SurfaceStartEnd[0,0]:self.SurfaceStartEnd[0,1],0]-1)
        J_LV = np.asarray(self.ETIndices[self.SurfaceStartEnd[0,0]:self.SurfaceStartEnd[0,1],1]-1)
        K_LV = np.asarray(self.ETIndices[self.SurfaceStartEnd[0,0]:self.SurfaceStartEnd[0,1],2]-1)
    
        # RV endo
        I_RV = np.asarray(self.ETIndices[self.SurfaceStartEnd[1,0]:self.SurfaceStartEnd[2,1],0]-1)
        J_RV = np.asarray(self.ETIndices[self.SurfaceStartEnd[1,0]:self.SurfaceStartEnd[2,1],1]-1)
        K_RV = np.asarray(self.ETIndices[self.SurfaceStartEnd[1,0]:self.SurfaceStartEnd[2,1],2]-1)
     
        # Epicardium     
        I_epi = np.asarray(self.ETIndices[self.SurfaceStartEnd[3,0]:self.SurfaceStartEnd[3,1],0]-1)
        J_epi = np.asarray(self.ETIndices[self.SurfaceStartEnd[3,0]:self.SurfaceStartEnd[3,1],1]-1)
        K_epi = np.asarray(self.ETIndices[self.SurfaceStartEnd[3,0]:self.SurfaceStartEnd[3,1],2]-1) 

        if surface == "all":    
            points3D=np.vstack((self.etPos[:,0],self.etPos[:,1],self.etPos[:,2])).T
            simplices=np.vstack((self.ETIndices[:,0]-1,self.ETIndices[:,1]-1,self.ETIndices[:,2]-1)).T
            tri_vertices=list(map(lambda index: points3D[index], simplices))

            triangles_LV=go.Mesh3d(
                    x = x,
                    y = y,
                    z = z,
                    color=face_color_LV,
                    i = I_LV,
                    j = J_LV,
                    k = K_LV,
                    opacity=1
                )
    
            triangles_RV=go.Mesh3d(
                    x = x,
                    y = y,
                    z = z,
                    color=face_color_RV,
                    name=my_name,
                    i = I_RV,
                    j = J_RV,
                    k = K_RV,
                    opacity=1
                )
    
            triangles_epi=go.Mesh3d(
                    x = x,
                    y = y,
                    z = z,
                    color=face_color_epi,
                    i = I_epi,
                    j = J_epi,
                    k = K_epi,
                    opacity=0.4
                )

            lists_coord=[[[T[k%3][c] for k in range(4)]+[ None]  for T in tri_vertices]  for c in range(3)]
            Xe, Ye, Ze=[functools.reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]
            
            #define the lines to be plotted
            lines=go.Scatter3d(
                            x=Xe,
                            y=Ye,
                            z=Ze,
                            mode='lines',
                            line=go.Line(color= 'rgb(0,0,0)', width=1.5),
                            showlegend=False
                )
                        
            return [triangles_epi, triangles_LV, triangles_RV, lines]
 
        if surface == "endo":
            points3D=np.vstack((self.etPos[:,0],self.etPos[:,1],self.etPos[:,2])).T
            simplices=np.vstack((self.ETIndices[self.SurfaceStartEnd[0,0]:self.SurfaceStartEnd[2,1],0]-1,self.ETIndices[self.SurfaceStartEnd[0,0]:self.SurfaceStartEnd[2,1],1]-1,self.ETIndices[self.SurfaceStartEnd[0,0]:self.SurfaceStartEnd[2,1],2]-1)).T
            tri_vertices=list(map(lambda index: points3D[index], simplices))
                                   
            triangles_LV=go.Mesh3d(
                    x = x,
                    y = y,
                    z = z,
                    color=face_color_LV,
                    i = I_LV,
                    j = J_LV,
                    k = K_LV,
                    opacity=opacity
                )
    
            triangles_RV=go.Mesh3d(
                    x = x,
                    y = y,
                    z = z,
                    color=face_color_RV,
                    name=my_name,
                    i = I_RV,
                    j = J_RV,
                    k = K_RV,
                    opacity=opacity
                )
            #define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle             
            lists_coord=[[[T[k%3][c] for k in range(4)]   for T in tri_vertices]  for c in range(3)]
            Xe, Ye, Ze=[functools.reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]
            
            #define the lines to be plotted
            lines=go.Scatter3d(
                            x=Xe,
                            y=Ye,
                            z=Ze,
                            mode='lines',
                            line=go.Line(color= 'rgb(0,0,0)', width=1.5),
                            showlegend=False
                )
                        
            return [triangles_LV, triangles_RV, lines]            

        if surface == "epi":
            points3D=np.vstack((self.etPos[:,0],self.etPos[:,1],self.etPos[:,2])).T
            simplices=np.vstack((self.ETIndices[self.SurfaceStartEnd[3,0]:self.SurfaceStartEnd[3,1],0]-1,self.ETIndices[self.SurfaceStartEnd[3,0]:self.SurfaceStartEnd[3,1],1]-1,self.ETIndices[self.SurfaceStartEnd[3,0]:self.SurfaceStartEnd[3,1],2]-1)).T
            tri_vertices=list(map(lambda index: points3D[index], simplices))
                   
            triangles_epi=go.Mesh3d(
                    x = x,
                    y = y,
                    z = z,
                    color=face_color_epi,
                    i = I_epi,
                    j = J_epi,
                    k = K_epi,
                    opacity=0.8
                )
    
            #define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle             
            lists_coord=[[[T[k%3][c] for k in range(4)]   for T in tri_vertices]  for c in range(3)]
            Xe, Ye, Ze=[functools.reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]
            
            #define the lines to be plotted
            lines=go.Scatter3d(
                            x=Xe,
                            y=Ye,
                            z=Ze,
                            mode='lines',
                            line=go.Line(color= 'rgb(0,0,0)', width=1.5),
                            showlegend=False
                )
                        
            return [triangles_epi, lines] 

    def SAXSliceShiffting(self):
        ''' Performs breath-hold misregistration correction when the dataset does not contain any LAX slices 
            a stiff linear least squares fit of the LV with D-Affine regularisation is performed to align a 3D LV model with the long axis (defined by mitral point, apex and tricuspid points) slices. 
            By using a very stiff fit, the overall shape is preserved. Intersections between the 3D model and the 2D SAX slices are then calculated and the 2D short axis (SAX) 
            slices are aligned with the intersection contours.
               Input: biventricular model
               Output: 2D Translations '''

        # Stiff LV fit
        # ---------------
        High_weight = 10E+7
        [full_matrix_coefficient_points, index, d, w_out, contourType, distance_d_prior] = self.CalcDataSetXi(1000)

        PriorPosition = np.dot(full_matrix_coefficient_points,self.control_mesh)

        w = np.eye(len(PriorPosition))
        for i in range(len(PriorPosition)):
            w[i,i] = w_out[i]

        WPG = np.dot(w,full_matrix_coefficient_points)                                                                
        GTPTWTWPG = np.dot(WPG.T,WPG)
        A = GTPTWTWPG + High_weight*(self.GTSTSG_x + self.GTSTSG_y + self.GTSTSG_z)
        Wd = np.dot(w,d-PriorPosition)
        rhs = np.dot(WPG.T,Wd)    
        sol = np.linalg.lstsq(A,rhs)
        solf = sol[0]

        saved_model = copy.deepcopy(self)
        saved_model.control_mesh = np.add(saved_model.control_mesh,solf)
        saved_model.etPos = np.dot(saved_model.matrix,saved_model.control_mesh)        
               
        disp = np.zeros((len(self.ImageOrientationPatient),2))
        
        index = np.max(self.DataSet.slice_number[(self.DataSet.ContourType == "saendocardialContour")])
        
        # Calculate intersection
        # -----------------------
        for i in range(len(self.ImageOrientationPatient)):
            
            # Check if there is an endocardial contour for the slice i.
            # -----------------------------------------------------------
            if len(self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour") & (self.DataSet.slice_number == i),:]) >0 :
  
                # Get the transformation from 2D to 3D
                # ----------------------------------------     
                Transformation = self.From2Dto3D(self.ImagePositionPatient[i],self.ImageOrientationPatient[i],i)
                
                # Get all the points on the slice i
                # ----------------------------------
                LV_points = self.DataSet.Evenly_spaced_points[(self.DataSet.ContourType == "saendocardialContour")& (self.DataSet.slice_number == i),:] 
                   
    
                pts_LV = np.ones((len(LV_points),4))
                pts_LV[:,0:3] = LV_points 
                Px_LV = np.dot(pts_LV,np.linalg.inv(Transformation).T)
                P2_LV = Px_LV[:,0:2] / (np.vstack((Px_LV[:,3],Px_LV[:,3]))).T
        
                # We convert the origin of the image in 2D
                # -----------------------------------------
                pts_ImagePosition = np.ones(4) 
                
                pts_ImagePosition[0:3] = self.ImagePositionPatient[i] 
                Px_ImagePosition = np.dot(pts_ImagePosition,np.linalg.inv(Transformation).T)
                P2_ImagePosition = Px_ImagePosition[0:2] / (np.vstack((Px_ImagePosition[3],Px_ImagePosition[3]))).T            
            
                # Give the intersection of the LV with the slices
                # -----------------------------------------------
                intersection_surface = saved_model.GetIntersectionWithDICOMImage(i)
                
                # If no intersection or last slice
                if (intersection_surface[0,0] == -1) or i == index:
                    
                    # Get centroid previous slice and displace this slices
                    # -----------------------------------------------------
                    # Revised by Lee. HCMR slice index overflow in previous code
                    if i-1<0:
                        intersection_surface = saved_model.GetIntersectionWithDICOMImage(i+1)
                        if (intersection_surface[0,0] == -1):
                            p = i+1
                            
                            intersection_surface = saved_model.GetIntersectionWithDICOMImage(p)
                            while intersection_surface[0,0] == -1:
                                p = p+1                            
                                intersection_surface = saved_model.GetIntersectionWithDICOMImage(p)
                        
                            intersection_surface = saved_model.GetIntersectionWithDICOMImage(p)
                    else:
                        intersection_surface = saved_model.GetIntersectionWithDICOMImage(i-1)
                    
                    # If there is no intersection for the next slice again, we look for the last one with an intersection
                        if (intersection_surface[0,0] == -1):
                            p = i-1
                            
                            intersection_surface = saved_model.GetIntersectionWithDICOMImage(p)
                            while intersection_surface[0,0] == -1:
                                p = p-1                            
                                intersection_surface = saved_model.GetIntersectionWithDICOMImage(p)
                        
                            intersection_surface = saved_model.GetIntersectionWithDICOMImage(p)

                    # Get triangulation
                    T = Delaunay(intersection_surface)
                    n = len(T.simplices)
                    W = np.zeros((n,1))
                    C=0   
                     
                    for k in range(n):
                        sp = intersection_surface[T.simplices[k,:],:]
                        a = np.linalg.norm(sp[1,:]-sp[0,:])
                        b = np.linalg.norm(sp[2,:]-sp[1,:])
                        c = np.linalg.norm(sp[2,:]-sp[0,:])
                        s = (a+b+c)/2
                        W[k] = np.sqrt(s*(s-a)*(s-b)*(s-c)) 
                        C = C + np.multiply(W[k],sp.mean(axis=0))

                    C = C/np.sum(W)  

                    centroid_LV_Data = P2_LV.mean(axis=0)
                    #print('no inter: ', C, centroid_LV_Data)

                    displacement = C - centroid_LV_Data 
                
                    disp[i,:] = displacement
                    
                    # Get all points slice
                    LV_points = self.DataSet.Evenly_spaced_points[(self.DataSet.slice_number == i),:]                     
                    pts_LV = np.ones((len(LV_points),4))

                    pts_LV[:,0:3] = LV_points 
                    Px_LV = np.dot(pts_LV,np.linalg.inv(Transformation).T)
                    P2 = Px_LV[:,0:2] / (np.vstack((Px_LV[:,3],Px_LV[:,3]))).T

                    for j in range(len(P2)):
                        P2[j,0] = P2[j,0] + displacement[0]
                        P2[j,1] = P2[j,1] + displacement[1]
                    
                    P2_ImagePosition = np.add(P2_ImagePosition,displacement)
                    
                    # We save new centroid
                    pts_centroid = np.array([centroid_LV_Data[0]+displacement[0],centroid_LV_Data[1]+displacement[1],0,1])
                    Px_centroid = np.dot(Transformation,pts_centroid.T)                    
                    
                    # Back to 3D   
                    pts_LV = np.ones((len(P2),4))
                    pts_LV[:,0:2] = P2
                    pts_LV[:,2] = [0]*len(P2)
                    pts_LV[:,3] = [1]*len(P2)
                    Px_LV = np.dot(Transformation,pts_LV.T)
                    P3_LV = Px_LV[0:3,:] / (np.vstack((Px_LV[3,:],np.vstack((Px_LV[3,:],Px_LV[3,:])))))                
                      
                    # Back to 3D ImagePositionPatient
                    pts_P2_ImagePosition = np.array([P2_ImagePosition[0,0],P2_ImagePosition[0,1],0,1])
                    Px_LV_ImagePosition = np.dot(Transformation,pts_P2_ImagePosition.T)

                    P3_LV_ImagePosition = Px_LV_ImagePosition[0:3] / (np.vstack((Px_LV_ImagePosition[3],np.vstack((Px_LV_ImagePosition[3],Px_LV_ImagePosition[3]))))).T 

                    
                    indexes = np.where(self.DataSet.slice_number == i) 
                    contType = self.DataSet.ContourType[indexes]

                    self.DataSet.Evenly_spaced_points = np.delete(self.DataSet.Evenly_spaced_points, indexes, 0)               
                    self.DataSet.Evenly_spaced_points = np.vstack((self.DataSet.Evenly_spaced_points,P3_LV.T))

                    self.DataSet.ContourType = np.delete(self.DataSet.ContourType, indexes, 0)       
                    self.DataSet.slice_number = np.delete(self.DataSet.slice_number, indexes, 0)  

                    self.DataSet.ContourType = np.hstack((self.DataSet.ContourType,contType))
                    self.DataSet.slice_number = np.hstack((self.DataSet.slice_number,[i]*np.asarray(indexes).shape[1]))             
                    
                else:
                    # find centroid 2D LV
                    T = Delaunay(intersection_surface)
                    n = len(T.simplices)
                    W = np.zeros((n,1))
                    C=0

                    for k in range(n):
                        sp = intersection_surface[T.simplices[k,:],:]
                        a = np.linalg.norm(sp[1,:]-sp[0,:])
                        b = np.linalg.norm(sp[2,:]-sp[1,:])
                        c = np.linalg.norm(sp[2,:]-sp[0,:])
                        s = (a+b+c)/2
                        W[k] = np.sqrt(s*(s-a)*(s-b)*(s-c)) 
                        C = C + np.multiply(W[k],sp.mean(axis=0))

                    C = C/np.sum(W)

                    # Get centroid
                    centroid_LV_Data = P2_LV.mean(axis=0)
                    #print('inter: ', C, centroid_LV_Data)
                    
                    displacement = C - centroid_LV_Data               
                    disp[i,:] = displacement
                    
                    # Get all points slice
                    LV_points = self.DataSet.Evenly_spaced_points[(self.DataSet.slice_number == i),:]                     
                    pts_LV = np.ones((len(LV_points),4))

                    pts_LV[:,0:3] = LV_points 
                    Px_LV = np.dot(pts_LV,np.linalg.inv(Transformation).T)
                    P2 = Px_LV[:,0:2] / (np.vstack((Px_LV[:,3],Px_LV[:,3]))).T

                    for j in range(len(P2)):
                        P2[j,0] = P2[j,0] + displacement[0]
                        P2[j,1] = P2[j,1] + displacement[1]
                        
                    pts_centroid = np.array([centroid_LV_Data[0]+displacement[0],centroid_LV_Data[1]+displacement[1],0,1])
                    Px_centroid = np.dot(Transformation,pts_centroid.T)                    
                    
                    # Back to 3D   
                    pts_LV = np.ones((len(P2),4))
                    pts_LV[:,0:2] = P2
                    pts_LV[:,2] = [0]*len(P2)
                    pts_LV[:,3] = [1]*len(P2)
                    Px_LV = np.dot(Transformation,pts_LV.T)
                    P3_LV = Px_LV[0:3,:] / (np.vstack((Px_LV[3,:],np.vstack((Px_LV[3,:],Px_LV[3,:])))))                

                    indexes = np.where(self.DataSet.slice_number == i) 
                    contType = self.DataSet.ContourType[indexes]

                    self.DataSet.Evenly_spaced_points = np.delete(self.DataSet.Evenly_spaced_points, indexes, 0)               
                    self.DataSet.Evenly_spaced_points = np.vstack((self.DataSet.Evenly_spaced_points,P3_LV.T))

                    self.DataSet.ContourType = np.delete(self.DataSet.ContourType, indexes, 0)       
                    self.DataSet.slice_number = np.delete(self.DataSet.slice_number, indexes, 0)  

                    self.DataSet.ContourType = np.hstack((self.DataSet.ContourType,contType))
                    self.DataSet.slice_number = np.hstack((self.DataSet.slice_number,[i]*np.asarray(indexes).shape[1]))

        return disp