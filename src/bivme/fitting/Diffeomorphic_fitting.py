import time
from cvxopt import matrix, solvers
from .build_model_tools import *
from plotly.offline import plot
import plotly.graph_objs as go
import os
import gzip
import shutil


def SolveProblemCVXOPT(
    biv_model, data_set, weight_GP, low_smoothing_weight, transmural_weight, txtfile
):
    """This function performs the proper diffeomorphic fit.
    Parameters
    ----------

    `case`  case id

    `weight_GP` data_points weight

    'low_smoothing_weight'  smoothing weight (for regularization term)

    Returns
    --------
        None
    """
    start_time = time.time()
    [
        data_points_index,
        w_out,
        distance_prior,
        projected_points_basis_coeff,
    ] = biv_model.compute_data_xi(weight_GP, data_set)
    # # exlude the outliers defined as (u-mean(u)) > 6*std(u)
    # projected_points_basis_coeff = projected_points_basis_coeff[abs(
    #     distance_prior - np.mean(distance_prior)) < 6 * np.std(distance_prior), :]
    # data_points_index = data_points_index[
    #     abs(distance_prior - np.mean(distance_prior)) < 6 * np.std(distance_prior)]
    # w_out = w_out[
    #     abs(distance_prior - np.mean(distance_prior)) < 6 * np.std(distance_prior)]
    #
    data_points = data_set.points_coordinates[data_points_index]
    prior_position = np.dot(projected_points_basis_coeff, biv_model.control_mesh)
    w = w_out * np.identity(len(prior_position))
    WPG = np.dot(w, projected_points_basis_coeff)
    GTPTWTWPG = np.dot(WPG.T, WPG)
    A = GTPTWTWPG + low_smoothing_weight * (
        biv_model.GTSTSG_x + biv_model.GTSTSG_y + transmural_weight * biv_model.GTSTSG_z
    )
    Wd = np.dot(w, data_points - prior_position)

    # print('low_smoothing_weight', low_smoothing_weight) ####to LOG

    # rhs = np.dot(WPG.T, Wd)
    previous_step_err = 0
    tol = 1e-6
    iteration = 0
    Q = 2 * A  # .T*A  # 2*A
    quadratic_form = matrix(0.5 * (Q + Q.T), tc="d")  # to make it symmetrical.
    prev_displacement = np.zeros((biv_model.numNodes, 3))
    step_err = np.linalg.norm(data_points - prior_position, axis=1)
    step_err = np.sqrt(np.sum(step_err) / len(prior_position))
    print("Explicitly constrained fit")

    while abs(step_err - previous_step_err) > tol and iteration < 10:
        print("     Iteration #" + str(iteration + 1) + " ECF error " + str(step_err))
        with open(txtfile, "a") as f:  # LDT
            f.write(
                "     Iteration #"
                + str(iteration + 1)
                + " Smoothing weight "
                + str(low_smoothing_weight)
                + "\t ECF error "
                + str(step_err)
                + "\n"
            )

        previous_step_err = step_err
        linear_part_x = matrix(
            (2 * np.dot(prev_displacement[:, 0].T, A) - 2 * np.dot(Wd[:, 0].T, WPG).T),
            tc="d",
        )
        linear_part_y = matrix(
            (2 * np.dot(prev_displacement[:, 1].T, A) - 2 * np.dot(Wd[:, 1].T, WPG).T),
            tc="d",
        )
        linear_part_z = matrix(
            (2 * np.dot(prev_displacement[:, 2].T, A) - 2 * np.dot(Wd[:, 2].T, WPG).T),
            tc="d",
        )

        linConstraints = matrix(generate_contraint_matrix(biv_model), tc="d")
        linConstraintNeg = -linConstraints
        G = matrix(np.vstack((linConstraints, linConstraintNeg)))
        size = 2 * (3 * len(biv_model.mBder_dx))
        bound = 1 / 3
        h = matrix([bound] * size)

        solvers.options["show_progress"] = False
        #  Solver: solvers.qp(P,q,G,h)
        #  see https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
        #  for explanation

        solx = solvers.qp(quadratic_form, linear_part_x, G, h)
        soly = solvers.qp(quadratic_form, linear_part_y, G, h)
        solz = solvers.qp(quadratic_form, linear_part_z, G, h)

        sx = [a for a in solx["x"]]  # LDT 15/11/21: this avoids .append()
        sy = [a for a in soly["x"]]
        sz = [a for a in solz["x"]]
        displacement = np.column_stack(
            (sx, sy, sz)
        )  # LDT 15/11/21: this avoids to repeat three times np.asarray

        """        solx = solvers.qp(quadratic_form, linear_part_x, G, h)
        soly = solvers.qp(quadratic_form, linear_part_y, G, h)
        solz = solvers.qp(quadratic_form, linear_part_z, G, h)
        sx = []
        sy = []
        sz = []
        for a in solx['x']:
            sx.append(a)
        for a in soly['x']:
            sy.append(a)
        for a in solz['x']:
            sz.append(a)
        displacement = np.zeros((biv_model.numNodes, 3))
        displacement[:, 0] = np.asarray(sx)
        displacement[:, 1] = np.asarray(sy)
        displacement[:, 2] = np.asarray(sz)"""
        # check if diffeomorphic
        Isdiffeo = biv_model.is_diffeomorphic(
            np.add(biv_model.control_mesh, displacement), 0.1
        )
        if Isdiffeo == 0:
            # Due to numerical approximations, epicardium and endocardium
            # can 'touch' (but not cross),
            # leading to a negative jacobian. If it happens, we stop.
            diffeo = 0
            print("Diffeomorphic condition not verified ")
            break
        else:
            prev_displacement[:, 0] = prev_displacement[:, 0] + sx
            prev_displacement[:, 1] = prev_displacement[:, 1] + sy
            prev_displacement[:, 2] = prev_displacement[:, 2] + sz
            biv_model.update_control_mesh(biv_model.control_mesh + displacement)

            prior_position = np.dot(
                projected_points_basis_coeff, biv_model.control_mesh
            )
            step_err = np.linalg.norm(data_points - prior_position, axis=1)
            step_err = np.sqrt(np.sum(step_err) / len(prior_position))
            iteration = iteration + 1
    with open(txtfile, "a") as f:  # LDT
        f.write("End of the implicitly constrained fit \n")
        f.write("--- %s seconds ---\n" % (time.time() - start_time))

    print("--- End of the explicitly constrained fit ---")
    print("--- %s seconds ---" % (time.time() - start_time))


def lls_fit_model(biv_model, weight_GP, data_set, smoothing_Factor):
    [
        index,
        weights,
        distance_prior,
        projected_points_basis_coeff,
    ] = biv_model.compute_data_xi(weight_GP, data_set)

    prior_position = np.linalg.multi_dot(
        [projected_points_basis_coeff, biv_model.control_mesh]
    )
    w = weights * np.identity(len(prior_position))

    WPG = np.linalg.multi_dot([w, projected_points_basis_coeff])
    GTPTWTWPG = np.linalg.multi_dot([WPG.T, WPG])
    # np.linalg.multi_dot faster than np.dot

    A = GTPTWTWPG + smoothing_Factor * (
        biv_model.GTSTSG_x + biv_model.GTSTSG_y + 0.001 * biv_model.GTSTSG_z
    )

    data_points_position = data_set.points_coordinates[index]
    Wd = np.linalg.multi_dot([w, data_points_position - prior_position])
    rhs = np.linalg.multi_dot([WPG.T, Wd])

    solf = np.linalg.solve(
        A.T.dot(A), A.T.dot(rhs)
    )  # solve the Moore-Penrose pseudo inversee
    err = np.linalg.norm(data_points_position - prior_position, axis=1)
    err = np.sqrt(np.sum(err) / len(prior_position))
    return solf, err


def MultiThreadSmoothingED(biv_model, weight_GP, data_set, txtfile):
    """This function performs a series of LLS fits. At each iteration the
    least squares optimisation is performed and the determinant of the
    Jacobian matrix is calculated.
    If all the values are positive, the subdivision surface is deformed by
    updating its control points, projections are recalculated and the
    regularization weight is decreased.
    As long as the deformation is diffeomorphic, smoothing weight is decreased.
        Input:
            case: case name
            weight_GP: data_points' weight
        Output:
            None. 'biv_model' is updated in the function itself
    """
    start_time = time.time()
    high_weight = weight_GP * 1e10  # First regularization weight
    isdiffeo = 1
    iteration = 1
    factor = 5
    min_jacobian = 0.1

    while (isdiffeo == 1) & (high_weight > weight_GP * 1e2) & (iteration < 50):
        # print('high_weight', high_weight) ####to LOG
        displacement, err = lls_fit_model(biv_model, weight_GP, data_set, high_weight)
        try:
            with open(txtfile, "a") as f:  # LDT
                f.write(
                    "     Iteration #"
                    + str(iteration)
                    + " Weight "
                    + str(high_weight)
                    + "\t ICF error "
                    + str(err)
                    + "\n"
                )
        except:
            pass

        print("     Iteration #" + str(iteration) + " ICF error " + str(err))
        isdiffeo = biv_model.is_diffeomorphic(
            np.add(biv_model.control_mesh, displacement), min_jacobian
        )
        if isdiffeo == 1:
            biv_model.update_control_mesh(np.add(biv_model.control_mesh, displacement))
            high_weight = (
                high_weight / factor
            )  # we divide weight by 'factor' and start again...
        else:
            # If Isdiffeo ==1, the model is not updated.
            # We divide factor by 2 and try again.
            if factor > 1:
                factor = factor / 2
                high_weight = high_weight * factor
                isdiffeo = 1
        iteration = iteration + 1
    try:
        with open(txtfile, "a") as f:  # LDT
            f.write("End of the implicitly constrained fit \n")
            f.write("--- %s seconds ---\n" % (time.time() - start_time))

    except:
        pass

    print("End of the implicitly constrained fit")
    print("--- %s seconds ---" % (time.time() - start_time))
    return high_weight


def generate_contraint_matrix(mesh):
    """This function generates constraints matrix to be given to cvxopt

    Parameters
    ----------

        mesh

    Returns
    --------

    constraints: constraints matrix

    """

    constraints = []
    for i in range(len(mesh.mBder_dx)):  # rows and colums will always be the same so
        # we just need to precompute this and then calculate the values...

        dXdxi = np.zeros((3, 3), dtype="float")

        dXdxi[:, 0] = np.dot(mesh.mBder_dx[i, :], mesh.control_mesh)
        dXdxi[:, 1] = np.dot(mesh.mBder_dy[i, :], mesh.control_mesh)
        dXdxi[:, 2] = np.dot(mesh.mBder_dz[i, :], mesh.control_mesh)

        g = np.linalg.inv(dXdxi)

        Gx = (
            np.dot(mesh.mBder_dx[i, :], g[0, 0])
            + np.dot(mesh.mBder_dy[i, :], g[1, 0])
            + np.dot(mesh.mBder_dz[i, :], g[2, 0])
        )
        constraints.append(Gx)

        Gy = (
            np.dot(mesh.mBder_dx[i, :], g[0, 1])
            + np.dot(mesh.mBder_dy[i, :], g[1, 1])
            + np.dot(mesh.mBder_dz[i, :], g[2, 1])
        )
        constraints.append(Gy)

        Gz = (
            np.dot(mesh.mBder_dx[i, :], g[0, 2])
            + np.dot(mesh.mBder_dy[i, :], g[1, 2])
            + np.dot(mesh.mBder_dz[i, :], g[2, 2])
        )
        constraints.append(Gz)

    return np.asmatrix(constraints)


def calc_smoothing_matrix_DAffine(model, e_weights, e_groups=None):
    """Changed by A.Mira to allow elements grouping with different
    weights.

    Parameters
    ----------
    `e_groups`  list of list with index of elements defining element group

    `e_weight`  nx3 array were n is the number of groups, defining the
    weights for each group of elements
    """

    # function that compiles the S'S matrix using D affine weights is a
    # 3 x1 vector containing the desired weight alog u, v and w
    # direction(element coordinates system)
    # Adaptation from calcDefSmoothingMatrix_RVLV3D.m
    if not model.build_mode:
        print(
            "To compute the smoothing matrix the model should be "
            "built with build_mode=True"
        )
        return
    if e_groups == None:
        e_groups = [list(range(model.numElements))]

    e_weights = np.array(e_weights)
    if np.isscalar(e_groups[0]):
        e_groups = [e_groups]
    if len(e_weights.shape) == 1:
        e_weights = np.array([e_weights])
    # Creation of Gauss points

    xig, wg = generate_gauss_points(4)
    ngt = xig.shape[0]
    nft = 3
    nDeriv = [0, 1, 5]
    # d / dxi1, d / dxi2, d / dxi3 in position  1, 2 and 6
    dxi = 0.01
    # step in x space for finite difference calculation
    dxi1 = np.concatenate(
        (np.ones((ngt, 1)) * dxi, np.zeros((ngt, 1)), np.zeros((ngt, 1))), axis=1
    )
    dxi2 = np.concatenate(
        (np.zeros((ngt, 1)), np.ones((ngt, 1)) * dxi, np.zeros((ngt, 1))), axis=1
    )
    dxi3 = np.concatenate(
        (np.zeros((ngt, 1)), np.zeros((ngt, 1)), np.ones((ngt, 1)) * dxi), axis=1
    )

    STSfull = np.zeros((model.numNodes, model.numNodes))
    Gx = np.zeros((model.numNodes, model.numNodes))
    Gy = np.zeros((model.numNodes, model.numNodes))
    Gz = np.zeros((model.numNodes, model.numNodes))

    dXdxi = np.zeros((3, 3))
    dXdxi11 = np.zeros((3, 3))
    dXdxi12 = np.zeros((3, 3))
    dXdxi21 = np.zeros((3, 3))
    dXdxi22 = np.zeros((3, 3))
    dXdxi31 = np.zeros((3, 3))
    dXdxi32 = np.zeros((3, 3))

    mBder = np.zeros((ngt, model.numNodes, 10))
    mBder11 = np.zeros((ngt, model.numNodes, 10))
    mBder12 = np.zeros((ngt, model.numNodes, 10))
    mBder21 = np.zeros((ngt, model.numNodes, 10))
    mBder22 = np.zeros((ngt, model.numNodes, 10))
    mBder31 = np.zeros((ngt, model.numNodes, 10))
    mBder32 = np.zeros((ngt, model.numNodes, 10))
    for et_index, et in enumerate(e_groups):
        weights = e_weights[et_index]
        for ne in et:
            nr = 0
            Sk = np.zeros(
                (3 * ngt * nft, model.numNodes, 3)
            )  # storage for smoothing arrays

            # gauss points ' basis functions

            for j in range(ngt):
                _, mBder[j, :, :], _ = model.evaluate_basis_matrix(
                    xig[j, 0], xig[j, 1], xig[j, 2], ne, 0, 0, 0
                )
                _, mBder11[j, :, :], _ = model.evaluate_basis_matrix(
                    xig[j, 0],
                    xig[j, 1],
                    xig[j, 2],
                    ne,
                    dxi1[j, 0],
                    dxi1[j, 1],
                    dxi1[j, 2],
                )
                _, mBder12[j, :, :], _ = model.evaluate_basis_matrix(
                    xig[j, 0],
                    xig[j, 1],
                    xig[j, 2],
                    ne,
                    -dxi1[j, 0],
                    -dxi1[j, 1],
                    -dxi1[j, 2],
                )
                _, mBder21[j, :, :], _ = model.evaluate_basis_matrix(
                    xig[j, 0],
                    xig[j, 1],
                    xig[j, 2],
                    ne,
                    dxi2[j, 0],
                    dxi2[j, 1],
                    dxi2[j, 2],
                )
                _, mBder22[j, :, :], _ = model.evaluate_basis_matrix(
                    xig[j, 0],
                    xig[j, 1],
                    xig[j, 2],
                    ne,
                    -dxi2[j, 0],
                    -dxi2[j, 1],
                    -dxi2[0, 2],
                )
                _, mBder31[j, :, :], _ = model.evaluate_basis_matrix(
                    xig[j, 0],
                    xig[j, 1],
                    xig[j, 2],
                    ne,
                    dxi3[j, 0],
                    dxi3[j, 1],
                    dxi3[j, 2],
                )
                _, mBder32[j, :, :], _ = model.evaluate_basis_matrix(
                    xig[j, 0],
                    xig[j, 1],
                    xig[j, 2],
                    ne,
                    -dxi3[j, 0],
                    -dxi3[j, 1],
                    -dxi3[j, 2],
                )

            # for all gauss pts ng
            for ng in range(ngt):
                # calculate dX / dxi at Gauss pt and surrounding.
                for nk, deriv in enumerate(nDeriv):
                    dXdxi[:, nk] = np.dot(mBder[ng, :, deriv], model.control_mesh)
                    dXdxi11[:, nk] = np.dot(mBder11[ng, :, deriv], model.control_mesh)
                    dXdxi12[:, nk] = np.dot(mBder12[ng, :, deriv], model.control_mesh)
                    dXdxi21[:, nk] = np.dot(mBder21[ng, :, deriv], model.control_mesh)
                    dXdxi22[:, nk] = np.dot(mBder22[ng, :, deriv], model.control_mesh)
                    dXdxi31[:, nk] = np.dot(mBder31[ng, :, deriv], model.control_mesh)
                    dXdxi32[:, nk] = np.dot(mBder32[ng, :, deriv], model.control_mesh)

                g = np.linalg.inv(dXdxi)
                g11 = np.linalg.inv(dXdxi11)
                g12 = np.linalg.inv(dXdxi12)
                g21 = np.linalg.inv(dXdxi21)
                g22 = np.linalg.inv(dXdxi22)
                g31 = np.linalg.inv(dXdxi31)
                g32 = np.linalg.inv(dXdxi32)
                h = np.zeros((3, 3, 3))
                h[:, :, 0] = (g11 - g12) / (2 * dxi)
                h[:, :, 1] = (g21 - g22) / (2 * dxi)
                h[:, :, 2] = (g31 - g32) / (2 * dxi)

                # 2 nd order derivatives[uu, uv, uw; uv, vv, vw; uw vw, ww]
                pindex = np.array([[3, 2, 7], [2, 4, 6], [7, 6, 8]])

                for nk in range(3):  # derivatives
                    for nj in range(nft):
                        try:
                            Sk[nr, :, nk] = wg[ng] * (
                                g[0, nj] * mBder[ng, :, pindex[nk, 0]]
                                + g[1, nj] * mBder[ng, :, pindex[nk, 1]]
                                + g[2, nj] * mBder[ng, :, pindex[nk, 2]]
                                + h[0, nj, nk] * mBder[ng, :, 0]
                                + h[1, nj, nk] * mBder[ng, :, 1]
                                + h[2, nj, nk] * mBder[ng, :, 5]
                            )
                        except:
                            print("stop")
                        nr = nr + 1

            STS1 = np.dot(Sk[:, :, 0].T, Sk[:, :, 0])
            STS2 = np.dot(Sk[:, :, 1].T, Sk[:, :, 1])
            STS3 = np.dot(Sk[:, :, 2].T, Sk[:, :, 2])

            STS = (weights[0] * STS1) + (weights[1] * STS2) + (weights[2] * STS3)
            Gx = Gx + weights[0] * STS1
            Gy = Gy + weights[1] * STS2
            Gz = Gz + weights[2] * STS3

            # stiffness matrix
            STSfull = STSfull + STS

    GTSTSG = STSfull  # I've already included G

    return GTSTSG, Gx, Gy, Gz


def plot_timeseries(dataset, folder, filename):
    fig = go.Figure(dataset[0][0])

    frames = [go.Frame(data=k[0], name=f"frame{k[1]}") for k in dataset[:]]

    updatemenus = [
        dict(
            buttons=[
                dict(
                    args=[
                        None,
                        {
                            "frame": {"duration": 200, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                    label="Play",
                    method="animate",
                ),
                dict(
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    label="Pause",
                    method="animate",
                ),
            ],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            type="buttons",
            x=0.21,
            xanchor="right",
            y=-0.075,
            yanchor="top",
        )
    ]

    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f"frame{k[1]}"],
                        dict(
                            mode="immediate",
                            frame=dict(duration=200, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label=f"frame{k[1]}",
                )
                for i, k in enumerate(dataset)
            ],
            # active=1,
            transition=dict(duration=0),
            x=0,  # slider starting position
            y=0,
            currentvalue=dict(
                font=dict(size=12), prefix="frame: ", visible=True, xanchor="center"
            ),
            len=1.0,
        )  # slider length
    ]

    # print(fig.data[0])

    # [print(list(filter(None, k['x']))) for k in fig.data ]

    min_x = np.min(
        [
            (np.min(list(filter(None, k["x"]))))
            for k in fig.data
            if len(list(filter(None, k["x"]))) > 0
        ]
    )
    min_y = np.min(
        [
            (np.min(list(filter(None, k["y"]))))
            for k in fig.data
            if len(list(filter(None, k["y"]))) > 0
        ]
    )
    min_z = np.min(
        [
            (np.min(list(filter(None, k["z"]))))
            for k in fig.data
            if len(list(filter(None, k["z"]))) > 0
        ]
    )

    max_x = np.max(
        [
            (np.max(list(filter(None, k["x"]))))
            for k in fig.data
            if len(list(filter(None, k["x"]))) > 0
        ]
    )
    max_y = np.max(
        [
            (np.max(list(filter(None, k["y"]))))
            for k in fig.data
            if len(list(filter(None, k["y"]))) > 0
        ]
    )
    max_z = np.max(
        [
            (np.max(list(filter(None, k["z"]))))
            for k in fig.data
            if len(list(filter(None, k["z"]))) > 0
        ]
    )

    # print('MinMax x', np.min(fig.data[0]['x']), np.max(fig.data[0]['x']))

    fig.update(frames=frames)
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=8, range=[round(min_x, -1) - 20, round(max_x, -1) + 20]),
            yaxis=dict(nticks=8, range=[round(min_y, -1) - 20, round(max_y, -1) + 20]),
            zaxis=dict(
                nticks=8,
                range=[round(min_z, -1) - 20, round(max_z, -1) + 20],
            ),
        ),
        scene_aspectmode="cube",
        updatemenus=updatemenus,
        sliders=sliders,
    )

    result = plot(
        fig,
        filename=os.path.join(folder, filename),
        auto_open=False,
        auto_play=False,
        include_plotlyjs="cdn",
    )
    """
        with open(os.path.join(folder,filename), 'rb') as f_in:
            with gzip.open(os.path.join(folder,filename+'.gz'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                shutil.remove(f_in)
        #return html
        """
