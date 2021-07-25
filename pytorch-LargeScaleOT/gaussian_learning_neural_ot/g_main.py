import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import time
from IPython.display import Image
from gaussian_learning_neural_ot.api.gaussian_functions import (gaussian_data_sampling_mu,
                                 gaussian_data_sampling_nu,
                                 nu_sampler_from_discrete_distr)

from gaussian_learning_neural_ot.api.constants import (f_net_default, u_net_default, v_vec_default,
                           centers_nu_default, cov_matrix_default,
                           c_cost_type_default, d_cost_type_default,
                           reg_modes_arr, batch_size_val_default,
                           reg_mode_default, eps_default,
                           epochs_default, batch_size_default,
                           dtype_default, device_default,
                           random_state_default, random_states_train_default,
                           mu_sampler_default, nu_data_val_default,
                           optimizer_mode_default, lr_default,
                           centers_nu_default,
                           dir_to_save_default)

from gaussian_learning_neural_ot.api.gaussian_functions import index_sampler

from gaussian_learning_neural_ot.api.neural_ot import Neural_OT
from gaussian_learning_neural_ot.api.neural_ot_continious_to_discrete import Neural_OT_continious_to_discrete


torch.manual_seed(42)

my_u_net = nn.Sequential(
                  nn.Linear(2, 200),
                  nn.ReLU(),
                  nn.Dropout(p = 0.5),
                  nn.Linear(200, 500),
                  nn.ReLU(),
                  nn.Dropout(p = 0.5),
                  nn.Linear(500, 500),
                  nn.ReLU(),
                  nn.Dropout(p = 0.5),
                  nn.Linear(500, 1)
                 )

my_f_net = nn.Sequential(
                  nn.Linear(2, 200),
                  nn.ReLU(),
                  nn.Dropout(p = 0.5),
                  nn.Linear(200, 500),
                  nn.ReLU(),
                  nn.Dropout(p = 0.5),
                  nn.Linear(500, 500),
                  nn.ReLU(),
                  nn.Dropout(p = 0.5),
                  nn.Linear(500, 2)
                 )

batch_size_val = 1000
# [batchsize,2]
nu_data = gaussian_data_sampling_nu(centers_nu_default,
                                                     cov_matrix_default,
                                                     batch_size_val,
                                                     random_state_default,
                                                     device = device_default)

my_v_vec = torch.zeros(batch_size_val,
                                   dtype = dtype_default)

my_eps = 1e-2
my_reg_mode = 'l2'
default_experiment = Neural_OT_continious_to_discrete(eps = my_eps, reg_mode = my_reg_mode)
default_experiment.replace_u(my_u_net)
default_experiment.replace_v(my_v_vec)

dual_loss_arr_batch = []
dual_loss_arr_val = []

lr = 1e-3
optimizer_mode = 'Adam'

epochs = 200
epoch_step_to_print = 20

random_states_train = range(epochs)

batch_size = 1000

default_experiment.stochastic_OT_computation(lr = lr, epochs = epochs,
                                             batch_size = batch_size,
                                             optimizer_mode = optimizer_mode,
                                             random_states_train = random_states_train,
                                             mu_sampler = mu_sampler_default,
                                             index_sampler = index_sampler,
                                             nu_data = nu_data,
                                             loss_arr_batch = dual_loss_arr_batch,
                                             loss_arr_val = dual_loss_arr_val,
                                             epoch_step_to_print = epoch_step_to_print)

plot_mode = 'dual'

default_experiment.plot_loss_graphs(loss_arr_batch = dual_loss_arr_batch,
                                    loss_arr_val = dual_loss_arr_val,
                                    optimizer_mode = optimizer_mode,
                                    plot_mode = plot_mode,
                                    lr = lr)


default_experiment.replace_f(my_f_net)

loss_arr_batch = []
loss_arr_val = []

lr = 1e-3
optimizer_mode = 'Adam'

epochs = 201
epoch_step_to_print = 10
random_states_train = range(epochs)
random_state_val = random_state_default

batch_size = 1000

default_experiment.optimal_map_learning(lr = lr, epochs = epochs,
                                             batch_size = batch_size,
                                             optimizer_mode = optimizer_mode,
                                             mu_sampler = mu_sampler_default,
                                             index_sampler = index_sampler,
                                             nu_data = nu_data,
                                             random_states_train = random_states_train,
                                             random_state_val = random_state_val,
                                             loss_arr_batch = loss_arr_batch,
                                             loss_arr_val = loss_arr_val,
                                             epoch_step_to_print = epoch_step_to_print,
                                             make_gif = True)

gif = default_experiment.create_path_to_gif(optimizer_mode, lr)
Image(gif)

plot_mode = 'mapping training'

default_experiment.plot_loss_graphs(loss_arr_batch = loss_arr_batch,
                                    loss_arr_val = loss_arr_val,
                                    optimizer_mode = optimizer_mode,
                                    plot_mode = plot_mode,
                                    lr = lr)

random_state = random_state_default

default_experiment.plot_2d_mapping_discrete_nu(mu_sampler = mu_sampler_default,
                                               nu_data_val = nu_data,
                                               random_state = random_state,
                                               show_plot = True)