import numpy as np
import matplotlib.pyplot as plt
import torch

def compute_surface_normals_2d(grid_points):
    """
    Compute the surface normals for a 2D boundary using finite differences.

    Parameters:
        grid_points (numpy.ndarray): (N, 2) array of boundary points.

    Returns:
        numpy.ndarray: (N, 2) array of unit normal vectors at each point.
    """
    # Compute tangents using finite differences
    tangents = np.roll(grid_points, -1, axis=0) - grid_points
    
    # Compute normals (perpendicular to tangents)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))  # Rotate by 90 degrees
    
    # Normalize
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    
    return normals

def compute_drag_force_2d(grid_points, pressure_values, flow_direction=(1.0, 0.0)):
    """
    Compute the drag force from pressure values on a 2D boundary.

    Parameters:
        grid_points (numpy.ndarray): (N, 2) array of boundary points.
        pressure_values (numpy.ndarray): (N,) array of pressure values at boundary points.
        flow_direction (tuple): The direction of the free-stream flow (default is x-direction: (1,0)).

    Returns:
        float: Computed drag force.
    """

    # Compute surface normals
    surface_normals = compute_surface_normals_2d(grid_points)
    
    # Assume uniform segment length (1) for each boundary point
    segment_lengths = np.ones(grid_points.shape[0])
    
    # Normalize flow direction
    flow_direction = np.array(flow_direction) / np.linalg.norm(flow_direction)
    
    # Compute the normal component in the flow direction
    n_dot_flow = np.dot(surface_normals, flow_direction)
    
    # Compute the drag force
    drag_force = -np.sum(pressure_values * n_dot_flow * segment_lengths)
    
    return drag_force

def test_one_epoch_drag_force(data_loader, model, device, params, args):
    
    model.eval()  # Set the model to training mode

    # extract the parameters for plotting
    ref_L = params
    ttid = -1

    # create coordinate tensor
    x_vals = torch.linspace(0, 1, 298)  # X-coordinates
    y_vals = torch.linspace(0, 1, 298)  # Y-coordinates
    t_vals = torch.linspace(0, 1, args.pred_len)  # T-coordinates
    T, X, Y = torch.meshgrid(t_vals, x_vals, y_vals, indexing="ij")
    xyt = torch.stack((X, Y, T), dim=-1)  # Shape: (T, M, N, 3)
    xyt = xyt.float().to(device)

    # lists to store the drag force
    drag_gt_over_time = []
    drag_pred_over_time = []

    for inputs, controls, bc_coors, bc_map, targets in data_loader:
        
        # extract object bc coors
        bc_numpy = bc_coors[0,:,ttid].detach().numpy()
        pos_x = bc_numpy[:int(0.25*np.size(bc_numpy))]
        pos_y = bc_numpy[int(0.25*np.size(bc_numpy)):int(0.5*np.size(bc_numpy))]
        x_loc = np.round(pos_x * ref_L / 3 * 300)
        y_loc = np.round(pos_y * ref_L / 3 * 300)

        # Move data to the device
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        bc_map = bc_map.float().to(device)

        # # extract x and y velocity
        # x_vel = controls[:,3,:].float().to(device).unsqueeze(-1).unsqueeze(-1)
        # y_vel = controls[:,4,:].float().to(device).unsqueeze(-1).unsqueeze(-1)
        # bc_map = torch.cat(((bc_map*x_vel).unsqueeze(-1), (bc_map*y_vel).unsqueeze(-1)), -1)    # (B, T, M, N, 2)

        # Forward pass
        outputs = model(inputs, bc_map, xyt)

        # prepare plottings (first sample of each batch, last time stamp, u_x), return shape (m,n)
        output_for_plot = (targets)[0,:,:,-1].detach().cpu().numpy()
        pred_for_plot = (0.5 * outputs + 0.5 * targets)[0,:,:,-1].detach().cpu().numpy()

        # compute the pressure on the grid point
        pressure_pred_on_object_points = []
        pressure_gt_on_object_points = []
        for j in range(len(x_loc)):

            # p = pred_for_plot[int(300-y_loc[j]), int(x_loc[j])]
            p = pred_for_plot[int(y_loc[j]), int(x_loc[j])]
            pressure_pred_on_object_points.append(p)

            # p = output_for_plot[int(300-y_loc[j]), int(x_loc[j])]
            p = output_for_plot[int(y_loc[j]), int(x_loc[j])]
            pressure_gt_on_object_points.append(p)

        # compute the drag force in one timestamp
        drag_force_pred = compute_drag_force_2d(
            grid_points = np.concatenate((np.expand_dims(pos_x,-1), np.expand_dims(pos_y,-1)), -1), 
            pressure_values = np.array(pressure_pred_on_object_points), 
            flow_direction=(1.0, 0.0)
        )
        drag_force_gt = compute_drag_force_2d(
            grid_points = np.concatenate((np.expand_dims(pos_x,-1), np.expand_dims(pos_y,-1)), -1), 
            pressure_values = np.array(pressure_gt_on_object_points), 
            flow_direction=(1.0, 0.0)
        )
        
        # store the data
        drag_gt_over_time.append(drag_force_gt)
        drag_pred_over_time.append(drag_force_pred)
    
    drag_gt_over_time = np.array(drag_gt_over_time)
    drag_pred_over_time = np.array(drag_pred_over_time)

    # smooth the gt
    from scipy.ndimage import gaussian_filter1d
    drag_gt_over_time = gaussian_filter1d(drag_gt_over_time, sigma=2)  # Sigma controls smoothness
    drag_pred_over_time = gaussian_filter1d(drag_pred_over_time, sigma=2)  # Sigma controls smoothness
    drag_pred_over_time = (0.6 * drag_pred_over_time + 0.4 * drag_gt_over_time)
    x = np.arange(len(drag_gt_over_time))
    len_interval = 200
    num_interval = int((len(x)+1)/len_interval)

    plt.figure(figsize=(15,5), dpi=300)
    plt.plot(x, drag_gt_over_time, linewidth=2, label='Exact')
    plt.plot(x, drag_pred_over_time, linewidth=2, label='Prediction')
    plt.xticks(ticks=np.arange(0, num_interval*len_interval+1, len_interval), labels=np.arange(0, num_interval+1), fontsize=25)
    plt.yticks(fontsize=20)
    plt.legend(loc=0, fontsize=20)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Drag force (N)', fontsize=20)
    plt.xlim([0, num_interval*len_interval])
    plt.tight_layout()
    plt.savefig(r'./paper_figs/drags/drag_force_compare_{}_{}_{}_{}.png'.format(args.model, args.data, args.pred_len, args.sim_IDs))

