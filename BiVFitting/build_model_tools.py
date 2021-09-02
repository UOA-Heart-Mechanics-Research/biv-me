import numpy as np

def generate_gauss_points(nb_points):
    '''
    Estimte gauss points and weight for x, y and z direction assuming
    interval(a,b) = [0,1]

    Parameters:
    ----------

    'nb_points' order of point scheme to be used

    Returns:
    -------

    `weights`  vector of gauss weights

    `points`  vector of gauss points
    '''
# cf Alistair's code

    nx=nb_points
    ny=nb_points
    nz=nb_points

    # set limits
    ax=0
    bx=1
    ay=0
    by=1
    az=0
    bz=1

    # get gauss points in x direction through the functions points_weights
    xgauss_weights, xgauss_points = gauss_points_weights(nx)
    # obtain actual values of the gauss points and weights
    xpoints=(bx-ax)*0.5*xgauss_points+(bx+ax)*0.5
    xweights=(bx-ax)*0.5*xgauss_weights

    # get y points in y direction through the functions points_weights
    ygauss_weights, ygauss_points =gauss_points_weights(ny)

    # obtain actual values of the gauss points and weights
    ypoints=(by-ay)*0.5*ygauss_points+(by+ay)*0.5
    yweights=(by-ay)*0.5*ygauss_weights

    # get z points in z direction through the functions points_weights
    zgauss_weights, zgauss_points =gauss_points_weights(nz)

    # obtain actual values of the gauss points and weights
    zpoints=(bz-az)*0.5*zgauss_points+(bz+az)*0.5
    zweights=(bz-az)*0.5*zgauss_weights

    xi = np.zeros((nx*ny*nz,3))
    weights = np.zeros(nx*ny*nz)
    count = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # position of approximating points xi[xs ys], nx2
                xi[count,0] = xpoints[i]
                xi[count,1] = ypoints[j]
                xi[count,2] = zpoints[k]
                weights[count] = xweights[i]*yweights[j]*zweights[k]
                count=count+1
    return xi, weights



def gauss_points_weights(number):
    ''' Generate gauss point and weights assuming interval[a,b] = [-1,1]

    Parameters:
    -----------

    `number` order of point scheme to be used

    Returns:
    --------

    `weights`  vector of gauss weights

    `points`  vector of gauss points

    '''

    zero_tolerance = 1 * 10e-10

    # obtain whether odd or even number of points desired
    odd_even = number%2

    if (odd_even == 1):
        loop = number - 1
    else:
        loop = number

    eta =np.zeros(int(loop / 2))
    w = np.zeros(int(loop/2))
    #loop over half number of points wanted
    for j  in range(int(loop *0.5)):
    # obtain initial estimate for first root
        eta[j] = np.cos(np.pi * (j+1 - 0.25) / (number + 0.5))
    # initialise delta to 1
        delta = 1
        while (abs(delta) > zero_tolerance):
            Pn, dPn = legendreCIM(number, eta[j])
            delta = -Pn / dPn
            eta[j] = eta[j] + delta
        w[j] = 2 / ((1 - eta[j] ** 2) * dPn ** 2)


    # record gauss points and weights
    points= np.zeros(number)
    weights = np.zeros(number)
    if (odd_even == 0):
        for j in range(int(loop *0.5)):
            points[j] = -eta[j]
            points[number-1 - j] = eta[j]
            weights[j] = w[j]
            weights[number -1 - j] = w[j]

    else:
        points[int(loop*0.5) ] = 0
        Pn,dPn = legendreCIM(number, points[int(loop*0.5 )])
        weights[int(loop*0.5) ] = 2 / ((1 - points[int(loop*0.5) ]**2) * dPn ** 2)
        for j  in range(int(loop *0.5)):
            points[j] = -eta[j]
            points[number-1 - j ] = eta[j]
            weights[j] = w[j]
            weights[number-1 - j ] = w[j]


    return weights,points



def legendreCIM(n, eta):
    '''Evaluates the Legendre polynomial and its derivative
    at a point eta(ith root)

    Parameters:
    ------------

    `n` number of roots / degrees of freedom

    `eta` current estimate of the root  i.e gauss  point

    Returns:
    --------

    `Pn`  value of polynomial at n

    `dPn` derivative of polynomial at n
    '''
    P= np.zeros(n+2)
    P[0] = 0
    P[1] = 1

    for i in range(1,n+1):
        P[i + 1] = ((2 * i - 1) * eta * P[i] - (i - 1) * P[i-1]) / i


    Pn = P[n+1 ]
    dPn = n * ((eta * P[n+1] - P[n]) / (eta ** 2 - 1))

    return Pn, dPn

def basis_function_bspline(s):
    ''' Evaluate the four uniform cubic B-Spline basis functions at a point s

    Parameters:
    -----------

    `s` float point where to evaluate the b-spline basis

    Returns:
    --------

    `bs` 4x1 vector b-spline basis functions value
    '''
    bs = np.zeros(4)
    bs[0] = (1/6) * (1-3*s+3*s*s-s*s*s)
    bs[1] = (1/6) * (4 - 6*s*s+3*s*s*s)
    bs[2] = (1/6) * (1+3*s+3*s*s-3*s*s*s)
    bs[3] = (1/6) * s*s*s

    return bs

def der2_basis_function_bspline(s):
    ''' Evaluate second derivatives of the four uniform cubic  B-Spline
        basis functions at point s

    Parameters:
    -----------

    `s` float point where to evaluate the b-spline basis

    Returns:
    --------

    `ds` 4x1 vector with second derivatives of the b-spline basis functions
    '''

    ds = np.zeros(4)
    ds[0] = 1-s
    ds[1] = 3*s-2
    ds[2] = 1-3*s
    ds[3] = s
    return ds

def der_basis_function_bspline(s):
    ''' Evaluate derivatives of the four uniform cubic  B-Spline
        basis functions at point s

    Parameters:
    -----------

    `s` float point where to evaluate the b-spline basis

    Returns:
    --------

    `ds` 4x1 vector with derivatives of the b-spline basis functions
    '''
    ds = np.zeros(4)
    ds[0] = -0.5*s*s + s - 0.5
    ds[1] = 1.5*s*s - 2*s
    ds[2] = -1.5*s*s + s + 0.5
    ds[3] = 0.5*s*s
    return ds


def adjust_boundary_weights(boundary, sWeights,tWeights):

    if int(boundary) &  1:
        tWeights[2] = tWeights[2] - tWeights[0]
        tWeights[1] = tWeights[1] + 2 * tWeights[0]
        tWeights[0] = 0


    if int(boundary) &  2:
        sWeights[1] = sWeights[1] - sWeights[3]
        sWeights[2] = sWeights[2] + 2 * sWeights[3]
        sWeights[3] = 0

    if int(boundary)& 4:
        tWeights[1] = tWeights[1] - tWeights[3]
        tWeights[2] = tWeights[2] + 2 * tWeights[3]
        tWeights[3] = 0


    if int(boundary) &  8:
        sWeights[2] = sWeights[2] - sWeights[0]
        sWeights[1] = sWeights[1] + 2 * sWeights[0]
        sWeights[0] = 0

    return sWeights, tWeights