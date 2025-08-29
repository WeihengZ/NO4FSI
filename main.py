import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import imageio
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import create_dataloaders
from utils import test_one_epoch_drag_force
from models.iterative import Iterate_model
from models.concat import Concat_model
from models.pooling import Pooling_model


# set arguments
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--data', type=str, default='random')
parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'plot'])
parser.add_argument('--model', type=str, default='concat')
parser.add_argument('--backend', type=str, default='Unet')
parser.add_argument('--train_method', type=str, default='adaptive')
parser.add_argument('--train_flag', type=str, default='retrain')
parser.add_argument('--pred_len', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--sample_freq', type=int, default=10)
parser.add_argument('--sim_IDs', type=str, default='[2,2]')
args = parser.parse_args()

# Set device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model and move it to the correct device
if args.model == 'iterative':
    model = Iterate_model(modes_x=20, modes_y=20, width=64, backend=args.backend)
if args.model == 'concat':
    model = Concat_model(args.pred_len, modes_x=20, modes_y=20, width=64, backend=args.backend)
if args.model == 'pooling':
    model = Pooling_model(modes_x=20, modes_y=20, width=64, backend=args.backend)


import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_one_sample(data_loader, args):
    for inputs, controls, bc_coors, bc_map, targets in data_loader:
        '''
        inputs: (1, M, N)
        bc_map: (1, T, M, N, 2)
        targets: (1, T, M, N)
        '''

        save_folder_path = r'./paper_figs/samples'

        # Ensure the save folder exists
        os.makedirs(save_folder_path, exist_ok=True)

        # Convert tensors to numpy arrays if needed
        if hasattr(inputs, "detach"):
            inputs = inputs.detach().cpu().numpy()
        if hasattr(bc_map, "detach"):
            bc_map = bc_map.detach().cpu().numpy()
        if hasattr(targets, "detach"):
            targets = targets.detach().cpu().numpy()

        # Plot inputs with fixed color range
        plt.figure(figsize=(8, 6))
        sns.heatmap(inputs[0, :, :, 0], cmap="viridis", cbar=True, vmin=-5, vmax=10)
        plt.title("Input")
        plt.savefig(os.path.join(save_folder_path, "input_sample.png"))
        plt.close()

        # Plot all boundary condition maps (color range will vary)
        for t in range(bc_map.shape[1]):  # Loop over time steps
            plt.figure(figsize=(8, 6))
            sns.heatmap(bc_map[0, t], cmap="viridis", cbar=True)
            plt.title(f"Boundary Condition Map - Timestep {t}")
            plt.savefig(os.path.join(save_folder_path, f"bc_map_sample_t{t}.png"))
            plt.close()

        # Plot all target maps with fixed color range
        for t in range(targets.shape[1]):  # Loop over time steps

            # extract object bc coors
            bc_numpy = bc_coors[0,:,t].detach().numpy()
            pos_x = bc_numpy[:int(0.25*np.size(bc_numpy))]
            pos_y = bc_numpy[int(0.25*np.size(bc_numpy)):int(0.5*np.size(bc_numpy))]
            x_loc = np.round(pos_x * ref_L / 3 * 300)
            y_loc = np.round(pos_y * ref_L / 3 * 300)

            plt.figure(figsize=(8, 6))
            sns.heatmap(targets[0, t, :, :, 0], cmap="viridis", cbar=True, vmin=-5, vmax=10)
            plt.scatter(x_loc, y_loc, s=0.5, color='k')
            plt.title(f"Target Map - Timestep {t}")
            plt.savefig(os.path.join(save_folder_path, f"target_map_sample_t{t}.png"))
            plt.close()

        assert 1 == 2  # Stop after the first batch


# Training function for one epoch
def train_one_epoch(data_loader, model, criterion, optimizer, device, args):
    model.train()  # Set the model to training mode
    epoch_loss = 0

    # create coordinate tensor
    x_vals = torch.linspace(0, 1, 298)  # X-coordinates
    y_vals = torch.linspace(0, 1, 298)  # Y-coordinates
    t_vals = torch.linspace(0, 1, args.pred_len)  # T-coordinates
    T, X, Y = torch.meshgrid(t_vals, x_vals, y_vals, indexing="ij")
    xyt = torch.stack((X, Y, T), dim=-1)  # Shape: (T, M, N, 3)
    xyt = xyt.float().to(device)

    for inputs, controls, bc_coors, bc_map, targets in data_loader:

        # random data sampler
        if np.random.rand() > 0.8:
            continue

        # Move data to the device
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        bc_map = bc_map.float().to(device)   # (B, T, M, N)

        # # extract x and y velocity
        # print(controls.shape)
        # x_vel = controls[:,3,:].float().to(device).unsqueeze(-1).unsqueeze(-1)  
        # y_vel = controls[:,4,:].float().to(device).unsqueeze(-1).unsqueeze(-1)  

        # print(bc_map.shape, x_vel.shape)
        # assert 1==2
        # bc_map = torch.cat(((bc_map*x_vel).unsqueeze(-1), (bc_map*y_vel).unsqueeze(-1)), -1)    # (B, T, M, N, 2)

        # Forward pass
        outputs = model(inputs, bc_map, xyt)

        # Compute loss
        if args.train_method == 'normal':
            loss = criterion(outputs, targets)
        else:
            tradeoff_coeff = torch.abs(targets - 
                torch.mean(torch.mean(targets,1),1).unsqueeze(1).unsqueeze(1))
            tradeoff_coeff = tradeoff_coeff / torch.amax(torch.amax(tradeoff_coeff, 1, keepdims=True), 2, keepdims=True)
            tradeoff_coeff = 1 + torch.sigmoid(tradeoff_coeff)
            loss = criterion(tradeoff_coeff * outputs, tradeoff_coeff * targets)

            # loss = torch.mean((outputs - targets)**2 / (targets)**2)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss for monitoring
        epoch_loss += loss.item()
    
    return epoch_loss

# validation function for one epoch
def val_one_epoch(data_loader, model, criterion, device, args):
    model.eval()  # Set the model to training mode
    u_loss = 0
    v_loss = 0
    p_loss = 0
    total_samples = 0

    # create coordinate tensor
    x_vals = torch.linspace(0, 1, 298)  # X-coordinates
    y_vals = torch.linspace(0, 1, 298)  # Y-coordinates
    t_vals = torch.linspace(0, 1, args.pred_len)  # T-coordinates
    T, X, Y = torch.meshgrid(t_vals, x_vals, y_vals, indexing="ij")
    xyt = torch.stack((X, Y, T), dim=-1)  # Shape: (T, M, N, 3)
    xyt = xyt.float().to(device)

    for inputs, controls, bc_coors, bc_map, targets in data_loader:
        # Move data to the device
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        bc_map = bc_map.float().to(device)

        # # extract x and y velocity
        # x_vel = controls[:,3,:].float().to(device).unsqueeze(-1).unsqueeze(-1)
        # y_vel = controls[:,4,:].float().to(device).unsqueeze(-1).unsqueeze(-1)
        # bc_map = torch.cat(((bc_map*x_vel).unsqueeze(-1), (bc_map*y_vel).unsqueeze(-1)), -1)    # (B, T, M, N, 2)

        num_sample = inputs.shape[0]
        # Forward pass
        outputs = model(inputs, bc_map, xyt)

        # Compute loss
        u_err = torch.mean(torch.abs(outputs[:,:,:,0] - targets[:,:,:,0])).detach().cpu().item()
        v_err = torch.mean(torch.abs(outputs[:,:,:,1] - targets[:,:,:,1])).detach().cpu().item()
        p_err = torch.mean(torch.abs(outputs[:,:,:,2] - targets[:,:,:,2])).detach().cpu().item()

        # Accumulate loss for monitoring
        u_loss += u_err
        v_loss += v_err
        p_loss += p_err
        total_samples += num_sample

    return u_loss / total_samples, v_loss / total_samples, p_loss / num_sample


# define function for testing and plotting
def test_one_epoch(data_loader, model, device, params, args):
    model.eval()  # Set the model to training mode

    # extract the parameters for plotting
    ref_L = params

    # create coordinate tensor
    x_vals = torch.linspace(0, 1, 298)  # X-coordinates
    y_vals = torch.linspace(0, 1, 298)  # Y-coordinates
    t_vals = torch.linspace(0, 1, args.pred_len)  # T-coordinates
    T, X, Y = torch.meshgrid(t_vals, x_vals, y_vals, indexing="ij")
    xyt = torch.stack((X, Y, T), dim=-1)  # Shape: (T, M, N, 3)
    xyt = xyt.float().to(device)

    sample_id = 0
    plot_set = set([150,350])
    for inputs, controls, bc_coors, bc_map, targets in data_loader:

        sample_id += 1

        if sample_id in plot_set:
        
            # extract object bc coors
            bc_numpy = bc_coors[0,:,-1].detach().numpy()
            pos_x = bc_numpy[:int(0.25*np.size(bc_numpy))]
            pos_y = bc_numpy[int(0.25*np.size(bc_numpy)):int(0.5*np.size(bc_numpy))]
            x_loc = np.round(pos_x * ref_L / 3 * 300)
            y_loc = np.round(pos_y * ref_L / 3 * 300)

            # Move data to the device
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            bc_map = bc_map.float().to(device)

            # Forward pass
            outputs = model(inputs, bc_map, xyt)
            # prepare plottings (first sample of each batch, last time stamp, u_x), return shape (m,n)

            # plot
            def make_a_plot(data, name):
                plt.figure(figsize=(6,5), dpi=300)
                cmap = 'coolwarm'
                ax = sns.heatmap(data, cmap=cmap, vmin=-1, vmax=3)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=15)  # Adjust fontsize here
                plt.scatter(x_loc, y_loc, s=0.5, color='k')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(r'./paper_figs/preds/{}_{}_{}.png'.format(name, sample_id, args.sim_IDs))
            
            def make_p_plot(data, name):
                plt.figure(figsize=(6,5), dpi=300)
                cmap = 'coolwarm'
                ax = sns.heatmap(data, cmap=cmap, vmin=-10, vmax=5)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=15)  # Adjust fontsize here
                plt.scatter(x_loc, y_loc, s=0.5, color='k')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(r'./paper_figs/preds/{}_{}_{}.png'.format(name, sample_id, args.sim_IDs))
            
            def make_err_plot(data, name):
                plt.figure(figsize=(6,5), dpi=300)
                cmap = 'coolwarm'
                ax = sns.heatmap(data, cmap=cmap, vmin=0, vmax=1)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=15)  # Adjust fontsize here
                plt.scatter(x_loc, y_loc, s=0.5, color='k')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(r'./paper_figs/preds/{}_{}_{}.png'.format(name, sample_id, args.sim_IDs))
            
            def make_p_err_plot(data, name):
                plt.figure(figsize=(6,5), dpi=300)
                cmap = 'coolwarm'
                ax = sns.heatmap(data, cmap=cmap,  vmin=0, vmax=4)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=15)  # Adjust fontsize here
                plt.scatter(x_loc, y_loc, s=0.5, color='k')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(r'./paper_figs/preds/{}_{}_{}.png'.format(name, sample_id, args.sim_IDs))

            make_a_plot(targets[0,:,:,0].detach().cpu().numpy(), 'u_gt')
            make_a_plot(outputs[0,:,:,0].detach().cpu().numpy(), 'u_pred')
            make_a_plot(targets[0,:,:,1].detach().cpu().numpy(), 'v_gt')
            make_a_plot(outputs[0,:,:,1].detach().cpu().numpy(), 'v_pred')
            make_p_plot(targets[0,:,:,2].detach().cpu().numpy(), 'p_gt')
            make_p_plot(outputs[0,:,:,2].detach().cpu().numpy(), 'p_pred')

            make_err_plot(
                np.abs(targets[0,:,:,0].detach().cpu().numpy() - outputs[0,:,:,0].detach().cpu().numpy()),
                'u_err')
            make_err_plot(
                np.abs(targets[0,:,:,1].detach().cpu().numpy() - outputs[0,:,:,1].detach().cpu().numpy()),
                'v_err')
            make_p_err_plot(
                np.abs(targets[0,:,:,2].detach().cpu().numpy() - outputs[0,:,:,2].detach().cpu().numpy()),
                'p_err')

# define a function to generating gif
def test_one_epoch_gif(data_loader, model, device, params, args):

    sim_id = eval(args.sim_IDs)[0]
    model.eval()  # Set the model to evaluation mode
    cmap = 'coolwarm'
    
    # Directory to store temporary frames
    frame_dir = "./res/frames/"
    os.makedirs(frame_dir, exist_ok=True)

    # List to store paths of all frames for the final GIF
    frame_files = []
    
    # Extract parameter for plotting
    ref_L = params

    # Initialize a global frame index counter
    frame_index = 0

    tt = -1

    # create coordinate tensor
    x_vals = torch.linspace(0, 1, 298)  # X-coordinates
    y_vals = torch.linspace(0, 1, 298)  # Y-coordinates
    t_vals = torch.linspace(0, 1, args.pred_len)  # T-coordinates
    T, X, Y = torch.meshgrid(t_vals, x_vals, y_vals, indexing="ij")
    xyt = torch.stack((X, Y, T), dim=-1)  # Shape: (T, M, N, 3)
    xyt = xyt.float().to(device)

    # Iterate over the data loader
    for batch_idx, (inputs, controls, bc_coors, bc_map, targets) in enumerate(data_loader):
        
        # Move data to the device
        inputs_ = inputs.float().to(device)
        targets = targets.float().to(device)
        bc_map = bc_map.float().to(device)

        # # extract x and y velocity
        # x_vel = controls[:,3,:].float().to(device).unsqueeze(-1).unsqueeze(-1)    # (B, T)
        # y_vel = controls[:,4,:].float().to(device).unsqueeze(-1).unsqueeze(-1)    # (B, T)
        # bc_map = torch.cat(((bc_map*x_vel).unsqueeze(-1), (bc_map*y_vel).unsqueeze(-1)), -1)    # (B, T, M, N, 2)
        
        # Forward pass
        outputs = model(inputs_, bc_map, xyt)

        # plot the last time step
        # Extract boundary coordinates for the current timestep
        bc_numpy = bc_coors[0, :, tt].detach().numpy()  # (M,)
        pos_x = bc_numpy[:int(0.25 * np.size(bc_numpy))]
        pos_y = bc_numpy[int(0.25 * np.size(bc_numpy)):int(0.5 * np.size(bc_numpy))]
        x_loc = np.round(pos_x * ref_L / 3 * 300)
        y_loc = np.round(pos_y * ref_L / 3 * 300)

        # Prepare plots for ground truth, prediction, and error at this time step
        output_for_plot = targets[0, :, :, 0].detach().cpu().numpy()
        pred_for_plot = outputs[0, :, :, 0].detach().cpu().numpy()
        abs_error = np.abs(output_for_plot - pred_for_plot)  # Absolute error

        vmin = np.amin(output_for_plot)
        vmax = np.amax(output_for_plot)

        # Create a new figure for each frame with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        
        # Ground Truth
        sns.heatmap(output_for_plot, cmap=cmap, ax=axes[0],  vmin=vmin, vmax=vmax, cbar=False)
        axes[0].scatter(x_loc, y_loc, s=0.5, color='k')
        axes[0].set_title(f'Ground Truth Displacement - Timestep {frame_index}')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Prediction
        sns.heatmap(pred_for_plot, cmap=cmap, ax=axes[1], vmin=vmin, vmax=vmax, cbar=False)
        axes[1].scatter(x_loc, y_loc, s=0.5, color='k')
        axes[1].set_title(f'Predicted Displacement - Timestep {frame_index}')
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        # Absolute Error
        sns.heatmap(abs_error, cmap='viridis', ax=axes[2], cbar=False)
        axes[2].scatter(x_loc, y_loc, s=0.5, color='k')
        axes[2].set_title(f'Absolute Error - Timestep {frame_index}')
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        # figure setting
        plt.tight_layout()

        # Save the frame with a globally unique index
        frame_file = os.path.join(frame_dir, f"frame_{frame_index:04d}.png")
        fig.savefig(frame_file)
        frame_files.append(frame_file)
        plt.close(fig)

        # Increment the global frame index
        frame_index += 1

    # Create a single GIF from all frames
    gif_path = f"./res/gifs/pred_vs_gt_combined_{args.model}_{args.data}_{args.pred_len}_{args.sample_freq}_{sim_id}.gif"
    with imageio.get_writer(gif_path, mode='I', duration=2) as writer:
        for frame_file in frame_files:
            writer.append_data(imageio.imread(frame_file))

    # Optionally, clean up the frame images after creating the GIF
    for frame_file in frame_files:
        os.remove(frame_file)

    print(f"Combined GIF saved at {gif_path}")

# define a function to compute drag force

# define a function for overall training and testing
def exp_main(loaders, model, optimizer, device, visual_freq, param_plot, args):

    # Define loss function and optimizer
    criterion = nn.MSELoss()

    # extract different data loaders
    train_loader, val_loader, test_loader, plot_loader = loaders

    '''
    Uncomment it to plot sample I-O pairs
    '''
    # plot_one_sample(test_loader, args)

    # model training
    if args.phase=='train':

        # train
        loss_term = train_one_epoch(train_loader, model, criterion, optimizer, device, args)

        epoch_err = val_one_epoch(val_loader, model, criterion, device, args)
        
        return loss_term, epoch_err
                    
    if args.phase=='test':
        # write results to the txt
        test_one_epoch_drag_force(plot_loader, model, device, param_plot, args)
        assert 1==2
        test_one_epoch(plot_loader, model, device, param_plot, args)
        
        epoch_mae = val_one_epoch(test_loader, model, criterion, device, args)
        with open(r"./res/errs/results_{}_{}.txt".format(args.model, args.data), "a") as file:  # Use 'a' to append instead of 'w'
            file.write(f"Model name: {args.model}, Sim_IDs: {args.sim_IDs}, data_name: {args.data}, pred_len: {args.pred_len}, 'sample freq:', {args.sample_freq}, test error: {epoch_mae}\n")
        
        return epoch_mae
    
    if args.phase=='plot':
        test_one_epoch(plot_loader, model, device, param_plot, args)
        # test_one_epoch_gif(plot_loader, model, device, param_plot, args)

# load the pretrained model
if args.train_flag == 'retrain':
    try:
        model.load_state_dict(torch.load(r'./res/trained_models/best_model_{}_{}_{}.pth'.format(args.model, args.data, args.pred_len), map_location="cpu", weights_only=True))
    except:
        print('No pre-trained model')
model = model.float().to(device)

# training loop over all the simulation
if args.phase == 'train':
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    min_avg_err = np.inf
    for ep in range(args.num_epochs):
        print('current epoch:', ep)
        avg_mae = 0
        num_eval = 0
        for SIM_ID in range(2,3):
            
            args.sim_IDs = '[{},{}]'.format(SIM_ID, SIM_ID)
            
            # Load data
            train_loader, val_loader, test_loader, plot_loader, ref_L =\
                create_dataloaders(args, bs=1)

            # train the model
            loaders = (train_loader, val_loader, test_loader, plot_loader)
            param_plot = ref_L
            train_loss, epoch_err = exp_main(loaders, model, optimizer, device, visual_freq=1, param_plot=param_plot, args=args)
            print('Simulation ID:', SIM_ID, 'loss:', train_loss)
            print('Error:', epoch_err)
            avg_mae += epoch_err[0]
            num_eval += 1
        
        print('validation average loss:', avg_mae / num_eval)
        if avg_mae < min_avg_err:
            min_avg_err = avg_mae
            torch.save(model.state_dict(), r'./res/trained_models/best_model_{}_{}_{}.pth'.format(args.model, args.data, args.pred_len))

if args.phase == 'test':

    avg_u = 0
    avg_v = 0
    avg_p = 0
    num_eval = 0
    optimizer = None
    for SIM_ID in range(5,6):
        
        args.sim_IDs = '[{},{}]'.format(SIM_ID, SIM_ID)
        
        # Load data
        train_loader, val_loader, test_loader, plot_loader, ref_L =\
            create_dataloaders(args, bs=1)

        # train the model
        loaders = (train_loader, val_loader, test_loader, plot_loader)
        param_plot = ref_L
        ue, ve, pe = exp_main(loaders, model, optimizer, device, visual_freq=1, param_plot=param_plot, args=args)
        print('Simulation ID:', SIM_ID, 'MSE err:', ue, ve, pe)
        avg_u += ue
        avg_v += ve
        avg_p += pe
        num_eval += 1

    print('Average error:', avg_u / num_eval, avg_v / num_eval, avg_p / num_eval)

if args.phase == 'plot':

    # Load data
    optimizer = None
    train_loader, val_loader, test_loader, plot_loader, ref_L =\
        create_dataloaders(args, bs=1)

    # train the model
    loaders = (train_loader, val_loader, test_loader, plot_loader)
    param_plot = ref_L
    exp_main(loaders, model, optimizer, device, visual_freq=1, param_plot=param_plot, args=args)
