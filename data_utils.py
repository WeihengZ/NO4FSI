import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.data import Data
from itertools import product
import numpy as np
# from torch_geometric.data import Data, Batch
import os

''' -------------------------------- For grid based model ------------------------------------- '''


def read_hdf5_object(data, ref):
    """Helper function to dereference an HDF5 object reference."""
    obj = data[ref]
    if isinstance(obj, h5py.Dataset):
        return np.array(obj)  # Convert to NumPy array if it is a dataset
    return obj  # Return the object as-is if not a dataset

def load_data(file_path='oscillating_diamondfoil_188.0Re_3.0Hz_0.15m_60deg_90phase.jld2'):

    # Open the .jld2 file
    with h5py.File(file_path, 'r') as data:
        # Explore the structure of the file
        # print("Keys:", list(data.keys()))

        # Load scalar t_hist (direct read since it's not an array)
        t_hist = data["t_hist"][()]
        dt = t_hist[1][0]
        num_t = t_hist[2]

        # Handle object references in other datasets
        x_hist = [read_hdf5_object(data, ref) for ref in data["x_hist"]]
        u_hist = [read_hdf5_object(data, ref) for ref in data["u_hist"]]
        p_hist = [read_hdf5_object(data, ref) for ref in data["p_hist"]]
        f_b_hist = [read_hdf5_object(data, ref) for ref in data["f_b_hist"]]
        bc_list = [read_hdf5_object(data, ref) for ref in data["bc_list"]]

        # # Print out the data to verify content
        # print(f"x_hist:", len(x_hist), x_hist[0].shape)
        # print(f"u_hist:", len(u_hist), u_hist[0].shape)
        # print(f"p_hist:", len(p_hist), p_hist[0].shape)
        # print(f"f_b_hist", len(f_b_hist), f_b_hist[0].shape)

    # Open the file and load relevant data
    with h5py.File(file_path, 'r') as data:
        # Load 'normalized_boundary'
        normalized_boundary = data['normalized_boundary'][()]
        
        # Extract 'nodes' from 'normalized_boundary'
        if isinstance(normalized_boundary, np.void) or isinstance(normalized_boundary, np.recarray):
            nodes = normalized_boundary['nodes']
        else:
            raise TypeError("Unexpected data type for normalized_boundary.")

        # print(f"Nodes: {nodes}")

        # Extract 'ref_L' and 'ref_u'
        ref_L = data['ref_L'][()]
        ref_u = data['ref_u'][()]

        # print(f"ref_L: {ref_L}, ref_u: {ref_u}")

        # Load 'f_b_hist' and dereference its elements
        f_b_hist = [read_hdf5_object(data, ref) for ref in data['f_b_hist']]
    
    return (dt, num_t), (x_hist, u_hist, p_hist, f_b_hist, bc_list), (ref_L, ref_u, nodes)

def create_BC_map(bound_coors, map_size, ref_L):

    # initialize the map with two components for each collocation point
    map = np.zeros(map_size+(2,))

    # extract object bc coors
    bc_numpy = bound_coors
    pos_x = bc_numpy[:int(0.25*np.size(bc_numpy))]
    pos_y = bc_numpy[int(0.25*np.size(bc_numpy)):int(0.5*np.size(bc_numpy))]
    vel_x = bc_numpy[int(0.5*np.size(bc_numpy)):int(0.75*np.size(bc_numpy))]
    vel_y = bc_numpy[int(0.75*np.size(bc_numpy)):]
    x_loc = np.round(pos_x * ref_L / 3 * 300)
    y_loc = np.round(pos_y * ref_L / 3 * 300)

    # initialize the map
    for i in range(len(x_loc)):
        map[int(x_loc[i]), int(y_loc[i]), 0] = vel_x[i]
        map[int(x_loc[i]), int(y_loc[i]), 1] = vel_y[i]
    
    return map

def extract_numpy_data(args, bs, sample_freq=10):

    # Extract configuration
    start_id, end_id = eval(args.sim_IDs)[0], eval(args.sim_IDs)[1]
    pred_len = args.pred_len
    phase = args.phase

    # Data containers
    SIM_all_train_inputs = []
    SIM_all_train_control_inputs = []
    SIM_all_bc_coor_inputs = []
    SIM_all_bc_maps_inputs = []
    SIM_all_train_outputs = []

    folder_path = '/projects/bbqg/wzhong/FSI/{}'.format(args.data)
    x_data = []
    u_data = []
    v_data = []
    p_data = []
    BC_coors = []
    BC_maps = []

    # Load and process data
    for i in range(start_id, end_id + 1):
        file_path = os.path.join(folder_path, f'oscillating_diamondfoil_{i}_188.0Re.jld2')
        (dt, num_t), (x_hist, u_hist, p_hist, f_b_hist, bc_list), (ref_L, ref_u, nodes) = load_data(file_path=file_path)

        # Restructure the data
        num_datapoints = len(x_hist)
        for j in range(num_datapoints):

            # high-level control signals
            loc = x_hist[j]

            # boundary location and velocity
            bound_coors = bc_list[j]

            # Reshape data of the fluid flow field
            ux = u_hist[j][:89700].reshape((299, 300))[:-1, 1:-1].T    # (298, 298)
            uy = u_hist[j][89700:].reshape((300, 299))[1:-1, :-1].T    # (298, 298)
            p = p_hist[j].reshape((300, 300))[1:-1, 1:-1].T         # (298, 298)

            # Create boundary condition map
            bc_map = create_BC_map(bound_coors, p.shape, ref_L).transpose(1, 0, 2)

            # Append data
            x_data.append(loc)
            u_data.append(ux)
            v_data.append(uy)
            p_data.append(p)
            BC_coors.append(bound_coors)
            BC_maps.append(bc_map)

    # Convert to numpy arrays
    x_data = np.array(x_data)
    u_data = np.array(u_data)
    v_data = np.array(v_data)
    p_data = np.array(p_data)
    BC_coors = np.array(BC_coors)
    BC_maps = np.array(BC_maps)

    # # Compute max and min for velocity and pressure
    # u_max, u_min = np.max(u_data), np.min(u_data)
    # v_max, v_min = np.max(v_data), np.min(v_data)
    # p_max, p_min = np.max(p_data), np.min(p_data)

    # # Normalize data
    # u_data = (u_data - u_min) / (u_max - u_min)
    # v_data = (v_data - v_min) / (v_max - v_min)
    # p_data = (p_data - p_min) / (p_max - p_min)

    return x_data, BC_coors, u_data, v_data, p_data, BC_maps, ref_L

class FSI_Dataset(Dataset):
    def __init__(self, 
        x_data, BC_coors, u_data, v_data, p_data, BC_maps,
        idx_list, pred_len, sample_freq, transform=None):
        """
        Args:
            data_paths (list): List of paths to data files.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.x_data = torch.from_numpy(x_data)   # (T, ...)
        self.BC_coors = torch.from_numpy(BC_coors)
        self.u_data = torch.from_numpy(u_data)
        self.v_data = torch.from_numpy(v_data)
        self.p_data = torch.from_numpy(p_data)
        self.BC_maps = torch.from_numpy(BC_maps)
        self.transform = transform
        self.pred_len = pred_len
        self.sample_freq = sample_freq
        self.idx_list = idx_list

    def __len__(self):
        # Return the number of samples
        return len(self.idx_list)
    
    def extract_data(self, data, idx):

        train_control_input = []
        for j in range(self.pred_len):
            datap = data[idx+(j+1)*self.sample_freq]
            train_control_input.append(datap.unsqueeze(1))
        train_control_input = torch.cat(tuple(train_control_input), 1)   # (B, T, M, N)

        return train_control_input
    
    def extract_bc(self, data, idx):

        train_control_input = []
        for j in range(self.pred_len):
            datap = data[idx+(j+1)*self.sample_freq]
            train_control_input.append(datap.unsqueeze(0))
        train_control_input = torch.cat(tuple(train_control_input), 0)   # (B, T, M, N)

        return train_control_input

    def __getitem__(self, idx):
        
        idxx = self.idx_list[idx]
        train_input = torch.cat((
            self.u_data[idxx, :, :].unsqueeze(-1), 
            self.v_data[idxx, :, :].unsqueeze(-1), 
            self.p_data[idxx, :, :].unsqueeze(-1)), -1).unsqueeze(0)    # (B, M, N, 3)
        train_control_input = self.extract_data(self.x_data, idxx).unsqueeze(0)    # (B, 6, T)
        train_bc_input = self.extract_data(self.BC_coors, idxx).unsqueeze(0)       # (B, 208, T)
        bc_map_input = self.extract_bc(self.BC_maps, idxx).unsqueeze(0)

        train_output = torch.cat((
            self.u_data[idxx+self.pred_len*self.sample_freq, :, :].unsqueeze(-1), 
            self.v_data[idxx+self.pred_len*self.sample_freq, :, :].unsqueeze(-1), 
            self.p_data[idxx+self.pred_len*self.sample_freq, :, :].unsqueeze(-1)), -1).unsqueeze(0)    # (B, M, N, 3)
        
        return train_input.squeeze(0), train_control_input.squeeze(0), train_bc_input.squeeze(0), bc_map_input.squeeze(0), train_output.squeeze(0)

def create_dataloaders(args, bs):

    # extract numpy data
    x_data, BC_coors, u_data, v_data, p_data, BC_maps, ref_L = extract_numpy_data(args, bs, args.sample_freq)

    # split the data
    num_total_samples = len(x_data) - args.pred_len * args.sample_freq - 1 
    all_idx_fix = np.arange(num_total_samples)
    all_idx = np.arange(num_total_samples)
    np.random.seed(0)
    np.random.shuffle(all_idx)
    num_train = int(0.6 * num_total_samples)
    num_val = int(0.1 * num_total_samples)
    train_idx = all_idx[:num_train]
    val_idx = all_idx[num_train:num_train+num_val]
    test_idx = all_idx[num_train+num_val:]

    # construct the dataset
    train_dataset = FSI_Dataset(x_data, BC_coors, 
                          u_data, v_data, p_data, BC_maps, 
                          train_idx, args.pred_len, args.sample_freq)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_dataset = FSI_Dataset(x_data, BC_coors, 
                          u_data, v_data, p_data, BC_maps, 
                          val_idx, args.pred_len, args.sample_freq)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_dataset = FSI_Dataset(x_data, BC_coors, 
                          u_data, v_data, p_data, BC_maps, 
                          test_idx, args.pred_len, args.sample_freq)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    plot_dataset = FSI_Dataset(x_data, BC_coors, 
                          u_data, v_data, p_data, BC_maps, 
                          all_idx_fix, args.pred_len, args.sample_freq)
    plot_loader = torch.utils.data.DataLoader(plot_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader, plot_loader, ref_L
    

