import torch
from torch import nn
import numpy as np
from functools import partial

from .gaussian_functions import (gaussian_data_sampling_mu, 
                                 gaussian_data_sampling_nu,
                                 nu_sampler_from_discrete_distr)

random_state_default = 42

eps_default = 1e-2
lr_default = 1e-3
reg_mode_default = 'l2'
reg_modes_arr = ['l2', 'entropy']
c_cost_type_default = 'l2'
d_cost_type_default = 'l2'
device_default = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype_default = torch.float64
epochs_default = 10
batch_size_default = 128
batch_size_val_default = 1024
optimizer_mode_default = 'Adam'

random_states_train_default = range(epochs_default)

scale = 2.
nu_centers = [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1./np.sqrt(2), 1./np.sqrt(2)],
            [1./np.sqrt(2), -1./np.sqrt(2)],
            [-1./np.sqrt(2), 1./np.sqrt(2)],
            [-1./np.sqrt(2), -1./np.sqrt(2)]
          ]
gaussian_num = len(nu_centers)
centers_nu_default = torch.Tensor([(scale*x,scale*y) for x,y in nu_centers])
center_mu_default = torch.zeros(2)
sigma = torch.Tensor([0.02])
init_cov_matrix = torch.eye(2)
cov_matrix_default = sigma*init_cov_matrix

mu_data_val_default = gaussian_data_sampling_mu(center_mu_default, 
                                                     cov_matrix_default, 
                                                     batch_size_val_default, 
                                                     random_state_default, 
                                                     device = device_default)
nu_data_val_default = gaussian_data_sampling_nu(centers_nu_default, 
                                                     cov_matrix_default, 
                                                     batch_size_val_default, 
                                                     random_state_default, 
                                                     device = device_default)

nu_sampler_dicrete_default = partial(nu_sampler_from_discrete_distr, device = device_default)
mu_sampler_default = partial(gaussian_data_sampling_mu, 
                                   center_mu = center_mu_default, 
                                   cov_matrix = cov_matrix_default, 
                                   device = device_default)
nu_sampler_default = partial(gaussian_data_sampling_nu, 
                                   centers_nu = centers_nu_default, 
                                   cov_matrix = cov_matrix_default, 
                                   device = device_default)



torch.manual_seed(random_state_default)
D_in = 2
D_out = 1
H = 128
u_net_default = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, 2*H),
    torch.nn.ReLU(),
    torch.nn.Linear(2*H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, 1),
)

v_net_default = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, 2*H),
    torch.nn.ReLU(),
    torch.nn.Linear(2*H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, 1),
)

f_D_out = D_in

f_net_default = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, 2*H),
    torch.nn.ReLU(),
    torch.nn.Linear(2*H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, f_D_out),
)

v_vec_default = torch.zeros(batch_size_val_default, 
                                   dtype = dtype_default)

plot_mapping_title_default = '1 and 8 gaussians'
dir_to_save_default = 'figures'
epoch_step_to_print_default = 50
