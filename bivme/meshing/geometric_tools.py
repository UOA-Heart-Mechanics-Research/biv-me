
import numpy as np
from scipy import optimize
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay


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
    c = np.linalg.lstsq(A,b,rcond=None)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    center = np.array([xc, yc])
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return center, r


def fit_elipse_2d(points, tolerance=0.01):
    """ This function fits a elipse to a set of 2D points
        Input:
            [x,y]: 2D points coordinates
            w: weights for points (optional)
        Output:
            [xc,yc]: center of the fitted elipsoid
            r: radius of the fitted elipsoid
    """

    (N, d) = np.shape(points)
    d = float(d)
    # Q will be our working array
    Q = np.vstack([np.copy(points.T), np.ones(N)])
    QT = Q.T

    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT, np.dot(np.linalg.inv(V),Q)))  # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u


    # center of the ellipse
    center = np.dot(points.T, u)
    # the A matrix for the ellipse
    A = np.linalg.inv(
        np.dot(points.T, np.dot(np.diag(u), points)) -
        np.array([[a * b for b in center] for a in center])) / d
    # Get the values we'd like to return
    U, s, rotation = np.linalg.svd(A)
    radii = 1.0 / np.sqrt(s)

    return center, radii, rotation


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


def LineIntersection(point_position, plane_orientation, P0, P1):
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

    R[0,0:3] = plane_orientation[0:3]
    R[1,0:3] = plane_orientation[3:6]
    R[2,0:3] = np.cross(R[0,0:3],R[1,0:3])
    R[3,0:3] = point_position

    normal = R[2,0:3]

    u = P1-P0

    nu = np.dot(normal,u)
    if np.all(nu==0): # orthogonal vectors u belongs to the plane
        return P0

    # compute how from P0 to reach the plane
    s = (np.dot(normal.T , (R[3,0:3] - P0))) / nu

    # compute P
    P = P0 + s * u

    return P


def generate_2Delipse_by_vectors(t, center, radii, rotation =None):
    """ This function generates points on elipse
        Input:
            t: point's angle on the circle
            v: small axis vector
            u: large axis vector
            r: radii, if scalar estimates an cirle
            C: center of the elipse
        Output:
            P_circle: points on ellipse/circle if r is scalar
    """
    if np.isscalar(radii):
        radii = [radii,radii]
    if rotation is None:
        rotation = np.array([[1,0],[0,1]])

    x = radii[0] * np.cos(t)
    y = radii[1] * np.sin(t)
    for i in range(len(x)):
            [x[i], y[i]] = np.dot([x[i], y[i]],rotation) + center
    return  np.array([x, y]).T


def apply_affine_to_points(affine_matrix, points_array):
    """ apply affine matrix to 3D points, only in-plane transformation is considered
        input:
            affine_matrix : 4x4 matrix describing the affine
                            transformation
            points_array: nx3 array with points coordinates
        output:
            y_points_array: nx2 array with points coordinate in the new
            position
     """
    points_array_4D = np.ones((len(points_array), 4))
    points_array_4D[:, 0:3] = points_array
    t_points_array = np.dot(points_array_4D, affine_matrix.T)
    t_points_array = t_points_array[:, 0:3] / (
        np.vstack((t_points_array[:, 3], t_points_array[:, 3], t_points_array[:, 3]))).T
    return t_points_array



def register_group_points_translation_only(source_points, target_points,
                                           weights = None):
    """ compute the optimal translation between two sets of grouped points
    echa group for the source points will be projected into the corresponding
    group from target points
    input:
        source_points = array of nx2 arrays with points coordinates, moving
                        points
        target_points = array of nx2 arrays with points coordinates,
                        fixed points
    output: 2D translation vector
    """
    if isinstance(source_points,list):
        source_points = np.array(source_points)
    if isinstance(target_points, list):
        target_points = np.array(target_points)

    if source_points.shape[0] != target_points.shape[0]:
        return np.array([0,0])

    def obj_function(x):
        f = 0
        nb = 0
        for index,target in enumerate(target_points):
            tree = cKDTree(target)
            new_points = source_points[index]+np.array(x)
            d, indx = tree.query(new_points, k=1, p=2)
            d[d>10] = 0
            nb = nb + len(d)
            if weights is None:
                f  = f+ sum(d)
            else:
                f = f + weights[index]*sum(d)
        return f/nb

    t = optimize.fmin(func=obj_function, x0=[0, 0],
                    disp=False)
    return t


def sort_consecutive_points(C):
    " add by A.Mira on 01/2020"
    if isinstance(C, list):
        C = np.array(C)
    Cx = C[0, :]
    lastP = Cx
    C_index = [0]
    index_list = np.array(range(1,C.shape[0]))
    Cr = np.delete(C, 0, 0)
    # iterate through points until all points are taken away
    while Cr.shape[0] > 0:
        # find the closest point from the last point at Cx
        i = (np.square(lastP - Cr)).sum(1).argmin()
        # remove that closest point from Cr and add to Cx
        lastP = Cr[i, :]
        Cx = np.vstack([Cx, lastP])
        C_index.append(index_list[i])
        Cr = np.delete(Cr, i, 0)
        index_list = np.delete(index_list,i)
    return C_index,Cx

def compute_area_weighted_centroid(points):

    # centroids were calculated using the area-weighted average
    # of the barycentre of the triangles from the triangulation
    # of the intersection points
    # Get triangulation
    T = Delaunay(points)
    n = len(T.simplices)
    W = np.zeros((n, 1))
    C = 0

    for k in range(n):
        sp = points[T.simplices[k, :], :]
        a = np.linalg.norm(sp[1, :] - sp[0, :])
        b = np.linalg.norm(sp[2, :] - sp[1, :])
        c = np.linalg.norm(sp[2, :] - sp[0, :])
        s = (a + b + c) / 2
        W[k] = np.sqrt(s * (s - a) * (s - b) * (s - c))
        C = C + np.multiply(W[k], sp.mean(axis=0))

    C = C / np.sum(W)

    return C

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens
    return arr

def project_points_on_line(a, b, points):
    ap = points-a
    ab = b-a
    result = a + np.array( [np.dot(x.reshape(1,-1),ab) for x in
                            ap])/np.dot(ab,ab) * ab
    return result


