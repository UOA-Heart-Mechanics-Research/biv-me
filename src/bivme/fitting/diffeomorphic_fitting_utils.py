import time
import cvxpy as cp
from .build_model_tools import *
from plotly.offline import plot
import plotly.graph_objs as go
import os
from bivme.fitting import BiventricularModel
from bivme.fitting import GPDataSet
import numpy as np
from loguru import logger
from copy import deepcopy
from .surface_enum import Surface
import scipy.sparse as sp

def solve_convex_problem(
    biv_model: BiventricularModel, data_set: GPDataSet, weight_gp: float, low_smoothing_weight: float, transmural_weight: float, my_logger: logger, collision_detection = False, model_prior : BiventricularModel = None
) -> float:
    """This function performs the proper diffeomorphic fit.
    Parameters
    ----------
    'biv_model': BiventricularModel instance
    'data_set': GPDataSet instance
    `case`: case id
    `weight_gp` data_points weight
    'low_smoothing_weight'  smoothing weight (for regularization term)
    'transmural_weight':  smoothing weight along the transmural direction (for regularization term)
    'output_file':  output file (where the errors are saved)
    Returns
    --------
        None
    """

    start_time = time.time()

    if collision_detection:
        data_points = model_prior.et_pos
        projected_points_basis_coeff = biv_model.basis_matrix
        w_out = weight_gp * np.ones((biv_model.et_pos.shape[0], 1))
    else:
        [
            data_points_index,
            w_out,
            _,
            projected_points_basis_coeff,
        ] = biv_model.compute_data_xi(weight_gp, data_set)

        data_points = data_set.points_coordinates[data_points_index]

    prior_position = np.dot(projected_points_basis_coeff, biv_model.control_mesh)
    w = w_out * np.identity(len(prior_position))
    WPG = np.dot(w, projected_points_basis_coeff)
    GTPTWTWPG = np.dot(WPG.T, WPG)

    A = GTPTWTWPG + low_smoothing_weight * (
        biv_model.gtstsg_x + biv_model.gtstsg_y + transmural_weight * biv_model.gtstsg_z
    )
    Wd = np.dot(w, data_points - prior_position)

    previous_step_err = 0
    tol = tol = 5e-4# 1e-6
    iteration = 0
    prev_displacement = np.zeros((biv_model.NUM_NODES, 3))
    step_err = np.linalg.norm(data_points - prior_position, axis=1)
    step_err = np.sqrt(np.sum(step_err) / len(prior_position))

    Q = 2 * A  # .T*A  # 2*A
    Q = sp.csc_matrix(Q + Q.T)  # Precompute symmetric matrix

    residuals = -1
    collision_iteration = 1

    solver_params = {'solver':'OSQP', 
                    'verbose':False, 
                    'max_iter':8000, 
                    'eps_abs':1e-3,       
                    'eps_rel':1e-3,   
                    'adaptive_rho':True,  # Can improve convergence    
                    'polish':False,
                    'warm_start':True}

    # precompute the G and h matrix
    size = 2 * (3 * len(biv_model.mbder_dx))
    bound = 1.0
    h = np.array([bound] * size)   

    G_rows = 6 * biv_model.mbder_dx.shape[0]

    G_param = cp.Parameter((G_rows, biv_model.NUM_NODES))
    h_param = cp.Parameter(G_rows)
    q_param = [cp.Parameter(biv_model.NUM_NODES) for _ in range(3)]

    variables = [cp.Variable(biv_model.NUM_NODES) for _ in range(3)]
    problems = [
        cp.Problem(cp.Minimize(0.5 * cp.quad_form(var, Q) + q.T @ var), [G_param @ var <= h_param])
        for q, var in zip(q_param, variables)
    ]

    while abs(step_err - previous_step_err) / (previous_step_err + 1e-8) > tol and iteration < 10 and collision_iteration < 3:
        my_logger.info(f"     Iteration {iteration + 1} Smoothing weight {low_smoothing_weight}	 ECF error {step_err}")

        previous_step_err = step_err

        linear_constraints = generate_contraint_matrix(biv_model)
        linear_constraints_neg = -linear_constraints

        G = np.vstack((linear_constraints, linear_constraints_neg)) * 3.0
        G = sp.csc_matrix(G)

        linear_part_x = 2 * np.dot(prev_displacement[:, 0].T, A) - 2 * np.dot(Wd[:, 0].T, WPG).T
        linear_part_y = 2 * np.dot(prev_displacement[:, 1].T, A) - 2 * np.dot(Wd[:, 1].T, WPG).T
        linear_part_z = 2 * np.dot(prev_displacement[:, 2].T, A) - 2 * np.dot(Wd[:, 2].T, WPG).T

        linear_part_x = sp.csc_matrix(linear_part_x.reshape(-1, 1))  # Column vector
        linear_part_y = sp.csc_matrix(linear_part_y.reshape(-1, 1))
        linear_part_z = sp.csc_matrix(linear_part_z.reshape(-1, 1))

        # Update the parameters
        G_param.value = G
        h_param.value = h   
        q_param[0].value = linear_part_x.toarray().flatten()
        q_param[1].value = linear_part_y.toarray().flatten()
        q_param[2].value = linear_part_z.toarray().flatten()

        # Solve all problems
        #t3 = time.time()
        for prob in problems:
            prob.solve(**solver_params, ignore_dpp=True)
        #print("Time taken to solve the optimization problem:", time.time() - t3)    

        # Combine results
        displacement = np.column_stack([var.value for var in variables])

        #print("Norm of disp:", np.linalg.norm(displacement))  
        # Adding ealry stopping condition to avoid unnecessary iterations
        if np.linalg.norm(displacement) < 1e-4:
            break

        # check if diffeomorphic
        Isdiffeo = biv_model.is_diffeomorphic(
            np.add(biv_model.control_mesh, displacement), 0.1
        )
        if Isdiffeo == False:
            # Due to numerical approximations, epicardium and endocardium
            # can 'touch' (but not cross),
            # leading to a negative jacobian. If it happens, we stop.
            break
        else:
            if collision_detection:
                updated_model = deepcopy(biv_model)
                updated_model.update_control_mesh(np.add(biv_model.control_mesh, displacement))
                current_collision = updated_model.detect_collision()
                inter = current_collision.difference(updated_model.reference_collision) 
                if bool(inter):
                    for surface in [Surface.RV_SEPTUM, Surface.RV_FREEWALL, Surface.RV_INSERT]:
                        surface_index = biv_model.get_surface_vertex_start_end_index(surface)
                        model_prior.et_pos[surface_index[0] : surface_index[1] + 1, :] = biv_model.et_pos[surface_index[0] : surface_index[1] + 1, :]
                    collision_iteration += 1
                    data_points = model_prior.et_pos
                    Wd = np.dot(w, data_points - prior_position)
                    step_err = -1

                else:
                    prev_displacement[:, 0] = prev_displacement[:, 0] + displacement[:, 0]
                    prev_displacement[:, 1] = prev_displacement[:, 1] + displacement[:, 1]
                    prev_displacement[:, 2] = prev_displacement[:, 2] + displacement[:, 2]
                    biv_model.update_control_mesh(biv_model.control_mesh + displacement)

                    prior_position = np.dot(
                        projected_points_basis_coeff, biv_model.control_mesh
                    )
                    step_err = np.linalg.norm(data_points - prior_position, axis=1)
                    step_err = np.sqrt(np.sum(step_err) / len(prior_position))
                    iteration = iteration + 1

            else:
                prev_displacement += displacement
                biv_model.update_control_mesh(biv_model.control_mesh + displacement)

                prior_position = np.dot(projected_points_basis_coeff, biv_model.control_mesh)
                step_err = np.linalg.norm(data_points - prior_position, axis=1)
                step_err = np.sqrt(np.sum(step_err) / len(prior_position))
                iteration += 1

    residuals = step_err

    my_logger.success(f"End of the implicitly constrained fit. Time taken: {time.time() - start_time}")
    return residuals

def fit_least_squares_model(biv_model: BiventricularModel, weight_gp: float, data_set: GPDataSet, smoothing_factor: float) -> [np.ndarray, float]:
    [
        index,
        weights,
        distance_prior,
        projected_points_basis_coeff,
    ] = biv_model.compute_data_xi(weight_gp, data_set)

    prior_position = np.linalg.multi_dot(
        [projected_points_basis_coeff, biv_model.control_mesh]
    )
    w = weights * np.identity(len(prior_position))

    w_pg = np.linalg.multi_dot([w, projected_points_basis_coeff])
    GTPTWTWPG = np.linalg.multi_dot([w_pg.T, w_pg])

    A = GTPTWTWPG + smoothing_factor * (
        biv_model.gtstsg_x + biv_model.gtstsg_y + 0.001 * biv_model.gtstsg_z
    )

    data_points_position = data_set.points_coordinates[index]
    wd = np.linalg.multi_dot([w, data_points_position - prior_position])
    rhs = np.linalg.multi_dot([w_pg.T, wd])

    solf = np.linalg.solve(
        A.T.dot(A), A.T.dot(rhs)
    )  # solve the Moore-Penrose pseudo inversee
    err = np.linalg.norm(data_points_position - prior_position, axis=1)
    err = np.sqrt(np.sum(err) / len(prior_position))
    return solf, err

def fit_least_squares_model_with_prior(biv_model: BiventricularModel, weight_gp: float, prior_model: BiventricularModel, smoothing_factor: float) -> [np.ndarray, float]:

    projected_points_basis_coeff = biv_model.basis_matrix
    data_points_position = prior_model.et_pos
    weights = weight_gp * np.ones((biv_model.et_pos.shape[0], 1))

    prior_position = np.linalg.multi_dot(
        [projected_points_basis_coeff, biv_model.control_mesh]
    )
    w = weights * np.identity(len(prior_position))

    w_pg = np.linalg.multi_dot([w, projected_points_basis_coeff])
    GTPTWTWPG = np.linalg.multi_dot([w_pg.T, w_pg])

    A = GTPTWTWPG + smoothing_factor * (
        biv_model.gtstsg_x + biv_model.gtstsg_y + 0.001 * biv_model.gtstsg_z
    )

    wd = np.linalg.multi_dot([w, data_points_position - prior_position])
    rhs = np.linalg.multi_dot([w_pg.T, wd])

    solf = np.linalg.solve(
        A.T.dot(A), A.T.dot(rhs)
    )  # solve the Moore-Penrose pseudo inversee
    err = np.linalg.norm(data_points_position - prior_position, axis=1)
    err = np.sqrt(np.sum(err) / len(prior_position))
    return solf, err

def solve_least_squares_problem(biv_model : BiventricularModel, weight_gp: float, data_set : GPDataSet, my_logger, collision_detection : bool = False, model_prior : BiventricularModel = None):
    """This function performs a series of LLS fits. At each iteration the
    least squares optimisation is performed and the determinant of the
    Jacobian matrix is calculated.
    If all the values are positive, the subdivision surface is deformed by
    updating its control points, projections are recalculated and the
    regularization weight is decreased.
    As long as the deformation is diffeomorphic, smoothing weight is decreased.
        Input:
            case: case name
            weight_gp: data_points' weight
        Output:
            None. 'biv_model' is updated in the function itself
    """
    start_time = time.time()
    high_weight = weight_gp * 1e10  # First regularization weight
    isdiffeo = 1
    iteration = 1
    factor = 5
    min_jacobian = 0.1
    collision_iteration = 1

    while (isdiffeo == 1) & (high_weight > weight_gp * 1e2) & (iteration < 50) & (collision_iteration < 3):

        if collision_detection:
            displacement, err = fit_least_squares_model_with_prior(biv_model, weight_gp, model_prior, high_weight)
        else:
            displacement, err = fit_least_squares_model(biv_model, weight_gp, data_set, high_weight)

        my_logger.info(f"     Iteration {iteration} Weight {high_weight}	 ICF error {err}")

        isdiffeo = biv_model.is_diffeomorphic(
            np.add(biv_model.control_mesh, displacement), min_jacobian
        )

        if isdiffeo == 1:
            if collision_detection:
                updated_model = deepcopy(biv_model)
                updated_model.update_control_mesh(np.add(biv_model.control_mesh, displacement))
                current_collision = updated_model.detect_collision()
                inter = current_collision.difference(updated_model.reference_collision) 
                if bool(inter):
                    # if there is a collision detected, we update the prior to the current RV shape and keep doing the fitting to get a closer LV shape to the original
                    # update prior
                    my_logger.info(f"Intersection detected. Fixing the RV")
                    for surface in [Surface.RV_SEPTUM, Surface.RV_FREEWALL, Surface.RV_INSERT]:
                        surface_index = biv_model.get_surface_vertex_start_end_index(surface)
                        model_prior.et_pos[surface_index[0]:surface_index[1] + 1, :] = biv_model.et_pos[surface_index[0]:surface_index[1]+1, :]

                    collision_iteration += 1

                else:
                    biv_model.update_control_mesh(np.add(biv_model.control_mesh, displacement))
                    high_weight = (
                        high_weight / factor
                    )  # we divide weight by 'factor' and start again...
            else:
                biv_model.update_control_mesh(np.add(biv_model.control_mesh, displacement))
                high_weight = (
                    high_weight / factor
                )  # we divide weight by 'factor' and start again...
        else:
            # If Isdiffeo !=1, the model is not updated.
            # We divide factor by 2 and try again.
            if factor > 1:
                factor = factor / 2
                high_weight = high_weight * factor
                isdiffeo = 1
        iteration += 1

    my_logger.success(f"End of the explicitely constrained fit. Time taken: {time.time() - start_time}")

    return high_weight

def generate_contraint_matrix(mesh):
    """
    Constraint matrix generator.
    Assumes:
        mesh.mbder_dx, mbder_dy, mbder_dz: (N, number_of_control_points)
        mesh.control_mesh: (number_of_control_points, 3)
    """

    mbdx, mbdy, mbdz = mesh.mbder_dx, mesh.mbder_dy, mesh.mbder_dz
    ctrl = mesh.control_mesh   # shape: (388, 3)
    N, K = mbdx.shape

    dXdxi = np.empty((N, 3, 3), dtype=np.float64)
    dXdxi[:, 0, :] = mbdx @ ctrl  # each row: (388,) @ (388,3) = (3,)
    dXdxi[:, 1, :] = mbdy @ ctrl
    dXdxi[:, 2, :] = mbdz @ ctrl

    g_inv = np.linalg.inv(dXdxi)  # shape (N, 3, 3)

    Gx = (
        g_inv[:, 0, 0][:, None] * mbdx +
        g_inv[:, 0, 1][:, None] * mbdy +
        g_inv[:, 0, 2][:, None] * mbdz
    )
    Gy = (
        g_inv[:, 1, 0][:, None] * mbdx +
        g_inv[:, 1, 1][:, None] * mbdy +
        g_inv[:, 1, 2][:, None] * mbdz
    )
    Gz = (
        g_inv[:, 2, 0][:, None] * mbdx +
        g_inv[:, 2, 1][:, None] * mbdy +
        g_inv[:, 2, 2][:, None] * mbdz
    )

    # Stack without vstack overhead
    out = np.empty((3 * N, K), dtype=np.float64)
    out[0::3, :] = Gx
    out[1::3, :] = Gy
    out[2::3, :] = Gz

    return out

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
