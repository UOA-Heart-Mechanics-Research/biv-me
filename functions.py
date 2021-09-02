import numpy as np
import plotly.graph_objs as go
from plotly import tools
#import plotly.plotly as py
from plotly.graph_objs import *
import plotly
import biventricularModel
import importlib
importlib.reload(biventricularModel)
from biventricularModel import *


# Auxiliary functions
def fit_circle_2d(x, y, w=[]):
    """ This function fits a circle to a set of 2D points
        Input:
            [x,y]: 2D points coordinates
            w: weights for points (optional)
        Output:
            [xc,yc]: center of the fitted circle
            r: radius of the fitted circle
    """

    x = np.array(x)
    y = np.array(y)
    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2
    
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = diag(w)
        A = np.dot(W,A)
        b = np.dot(W,b)
    
    # Solve by method of least squares
    c = np.linalg.lstsq(A,b)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

def rodrigues_rot(P, n0, n1):
    """ This function rotates data based on a starting and ending vector. Rodrigues rotation is used
        to project 3D points onto a fitting plane and get their 2D X-Y coords in the coord system of the plane
        Input:
            P: 3D points
            n0: plane normal
            n1: normal of the new XY coordinates system
        Output:
            P_rot: rotated points

    """
    # If P is only 1d np.array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0,n1)
    k = k/np.linalg.norm(k)
    theta = np.arccos(np.dot(n0,n1))
    
    # Compute rotated points
    P_rot = np.zeros((len(P),3))
    for i in range(len(P)):
        P_rot[i] = P[i]*np.cos(theta) + np.cross(k,P[i])*np.sin(theta) + k*np.dot(k,P[i])*(1-np.cos(theta))

    return P_rot

def Plot2DPoint(points, color_markers, size_markers,nameplot = " "):
    """ Plot 2D points 
        Input: 
            points: 2D points
            color_markers: color of the markers 
            size_markers: size of the markers 
            nameplot: plot name (default: " ")

        Output:
            trace: trace for figure
    """
    trace = go.Scatter(
             x=points[:,0],
             y=points[:,1],
             name = nameplot,
             mode='markers',
             marker=dict(size=size_markers,opacity=1.0,color = color_markers)
            )
    return [trace]

def Plot3DPoint(points, color_markers, size_markers,nameplot = " "):
    """ Plot 3D points
        Input: 
            points: 3D points
            color_markers: color of the markers 
            size_markers: size of the markers 
            nameplot: plot name (default: " ")

        Output:
            trace: trace for figure
    """

    trace = go.Scatter3d(
             x=points[:,0],
             y=points[:,1],
             z=points[:,2],
             name = nameplot,
             mode='markers',
             marker=dict(size=size_markers,opacity=1.0,color = color_markers)
            )
    return [trace]

def LineIntersection(ImageOrientationPatient,ImagePositionPatient,P0,P1):
    """ Find the intersection between line P0-P1 with the MRI image.
        Input:  
            P0 and P1 are both single vector of 3D coordinate points.
        Output: 
            P is the intersection point (if any, see below) on the image plane.
            P in 3D coordinate. Use M.PatientToImage for converting it into 2D coordinate.
                
        P will return empty if M is empty.
        P will also return empty if P0-P1 line is parallel with the image plane M.
        Adpted from Avan Suinesiaputra
    """

    R = np.identity(4)

    R[0,0:3] = ImageOrientationPatient[0:3]
    R[1,0:3] = ImageOrientationPatient[3:6]
    R[2,0:3] = np.cross(R[0,0:3],R[1,0:3])
    R[3,0:3] = ImagePositionPatient

    normal = R[2,0:3]

    u = P1-P0

    nu = np.dot(normal,u)

    # compute how from P0 to reach the plane
    s = (np.dot(normal.T , (R[3,0:3] - P0))) / nu

    # compute P
    P = P0 + s * u

    return P


def generate_circle_by_vectors(t, C, r, n, u):
    """ This function generates points on circle
        Input:
            t: point's angle on the circle
            n: normal vector
            u: normal vector            
        Output:
            P_circle: points on circle
    """
    n = n/np.linalg.norm(n)
    u = u/np.linalg.norm(u)
    P_circle = r*np.cos(t)[:,np.newaxis]*u + r*np.sin(t)[:,np.newaxis]*np.cross(n,u) + C
    return P_circle
