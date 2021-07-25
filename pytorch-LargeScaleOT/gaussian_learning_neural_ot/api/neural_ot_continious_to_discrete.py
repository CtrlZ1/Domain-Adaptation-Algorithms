import torch
import time
import copy
import os
import matplotlib.pyplot as plt 

from .gaussian_functions import index_sampler

from .constants import (f_net_default, u_net_default, v_vec_default,
			c_cost_type_default, d_cost_type_default,
			reg_modes_arr,
			reg_mode_default, eps_default,
			epochs_default, batch_size_default, 
			dtype_default, device_default,
			random_state_default, random_states_train_default,
			mu_sampler_default, nu_data_val_default, 
			optimizer_mode_default, lr_default,
			plot_mapping_title_default,
			dir_to_save_default,
			epoch_step_to_print_default)
			
from .neural_ot import Neural_OT

class Neural_OT_continious_to_discrete(Neural_OT):
    
    def __init__(self, f_net = f_net_default, u_net = u_net_default, v_vec = v_vec_default, 
                 reg_mode = reg_mode_default, eps = eps_default, 
                 dtype = dtype_default, device = device_default):

        Neural_OT.__init__(self, f_net = f_net, reg_mode = reg_mode, eps = eps, 
                 c_cost_type = c_cost_type_default,
                 d_cost_type = d_cost_type_default,
                 dtype = dtype_default, device = device_default)

        u_copy = copy.deepcopy(u_net)
        self.u = u_copy.to(device)
        copy_v = v_vec.clone()
        copy_v.requires_grad_
        self.v = copy_v.to(device)
        
    def replace_u(self, u):
        u_copy = copy.deepcopy(u)
        self.u = u_copy.to(self.device)
        
    def replace_v(self, v):
        copy_v = v.clone()
        copy_v.requires_grad_
        self.v = copy_v.to(self.device)
        
    def stochastic_OT_computation(self, epochs = epochs_default, batch_size = batch_size_default,
                                  random_state_val = random_state_default,
                                  random_states_train = random_states_train_default,
                                  mu_sampler = mu_sampler_default, 
                                  index_sampler = index_sampler,
                                  nu_data = nu_data_val_default,
                                  optimizer_mode = optimizer_mode_default, 
                                  lr = lr_default,
                                  loss_arr_batch = [],
                                  loss_arr_val = [],
                                  epoch_step_to_print = epoch_step_to_print_default):
        if (self.v.shape[0] != nu_data.shape[0]):
            raise ValueError("Vector v and nu_data should be the same size!")

        trainable_params = list(self.u.parameters()) + [self.v]
        
        if optimizer_mode == 'Adam':
            optimizer = torch.optim.Adam(trainable_params, lr = lr)
        elif optimizer_mode == 'SGD':
            optimizer = torch.optim.SGD(trainable_params, lr = lr)

        for epoch in range(epochs):

            start_time = time.time()

            # [batchsize,2]
            x_batch = mu_sampler(random_state = random_states_train[epoch], batch_size = batch_size)

            
            indexes_to_choice = index_sampler(nu_data_shape = nu_data.shape[0], 
                                              batch_size = batch_size, 
                                              random_state = random_states_train[epoch], 
                                              device = self.device)

            y_batch = nu_data[indexes_to_choice, :]
            u_batch = (self.u)(x_batch)
            v_batch = (self.v)[indexes_to_choice]
            
            loss_batch = self.dual_OT_loss_estimation(u_batch, v_batch, x_batch, y_batch)
            
            optimizer.zero_grad()

            loss_batch.backward()
            optimizer.step()


            end_time = time.time()
            consumed_time = end_time - start_time

            loss_batch_maximization = -loss_batch.item()
            
            x_batch_val = mu_sampler(random_state = random_state_val, batch_size = nu_data.shape[0])
            y_batch_val = nu_data
            
            self.u.eval()
            self.v.requires_grad_(False)
            
            u_batch_val = (self.u)(x_batch_val)
            v_batch_val = self.v
            
            loss_val = self.dual_OT_loss_estimation(u_batch_val, v_batch_val, 
                                                         x_batch_val, y_batch_val)
            
            loss_val_maximization = -loss_val.item()
            
            if (epoch % epoch_step_to_print == 0):
            	print("------------------------------")
            	print(f"Epoch_num = {epoch + 1}")
            	print(f"Consumed time = {consumed_time} seconds")
            	print(f"Loss estimation on sampled data = {loss_batch_maximization}")
            	print(f"Loss estimation on validation data = {loss_val_maximization}")

            loss_arr_batch.append(loss_batch_maximization)
            loss_arr_val.append(loss_val_maximization)
        
    def optimal_map_learning(self, epochs = epochs_default, batch_size = batch_size_default,
                                  random_state_val = random_state_default,
                                  random_states_train = random_states_train_default,
                                  mu_sampler = mu_sampler_default, 
                                  index_sampler = index_sampler,
                                  nu_data = nu_data_val_default,
                                  optimizer_mode = optimizer_mode_default, 
                                  lr = lr_default,
                                  loss_arr_batch = [],
                                  loss_arr_val = [],
                                  make_gif = False,
                                  dir_to_save = dir_to_save_default,
                                  random_state_plot = random_state_default,
                                  epoch_step_to_print = epoch_step_to_print_default):
        
        if (self.v.shape[0] != nu_data.shape[0]):
            raise ValueError("Vector v and nu_data should be the same size!")
                                  
        if (make_gif and not os.path.exists(dir_to_save)):
            os.mkdir(dir_to_save)                          
        
        trainable_params = list(self.f_net.parameters())
        
        if optimizer_mode == 'Adam':
            optimizer = torch.optim.Adam(trainable_params, lr = lr)
        elif optimizer_mode == 'SGD':
            optimizer = torch.optim.SGD(trainable_params, lr = lr)
            

        for epoch in range(epochs):

            start_time = time.time()
            
            x_batch = mu_sampler(random_state = random_states_train[epoch], batch_size = batch_size)
            #print(x_batch.device)
            
            indexes_to_choice = index_sampler(nu_data_shape = nu_data.shape[0], 
                                              batch_size = batch_size, 
                                              random_state = random_states_train[epoch], 
                                              device = self.device)
            y_batch = nu_data[indexes_to_choice, :]
            u_batch = (self.u)(x_batch)
            v_batch = (self.v)[indexes_to_choice]
            
            self.f_net.train()
            map_batch = (self.f_net)(x_batch)
            
            loss_batch = self.mapping_OT_loss_estimation(u_batch, v_batch, x_batch, y_batch, map_batch)
            
            optimizer.zero_grad()

            loss_batch.backward()
            optimizer.step()


            end_time = time.time()
            consumed_time = end_time - start_time

            loss_batch = loss_batch.item()
            
            x_batch_val = mu_sampler(random_state = random_state_val, batch_size = nu_data.shape[0])
            y_batch_val = nu_data
            
            u_batch_val = (self.u)(x_batch_val)
            v_batch_val = self.v
            
            self.f_net.eval()
            map_batch = (self.f_net)(x_batch)
            
            loss_val = self.mapping_OT_loss_estimation(u_batch, v_batch, x_batch, y_batch, map_batch)
            
            loss_val = loss_val.item()
            if (epoch % epoch_step_to_print == 0):
            	print("------------------------------")
            	print(f"Epoch_num = {epoch + 1}")
            	print(f"Consumed time = {consumed_time} seconds")
            	print(f"Loss estimation on sampled data = {loss_batch}")
            	print(f"Loss estimation on validation data = {loss_val}")
            	if make_gif:
            	    plot_mapping_title = plot_mapping_title_default + \
            	          fr', optimizer = {optimizer_mode}, $lr = {lr}$, epoch = {epoch + 1}, regularization type = {self.reg_mode}, $\varepsilon = {self.eps}$'
            	    name_fig = f'gaussians_optimizer_{optimizer_mode}_lr_{lr}_epoch_{epoch + 1}_regularization_type_{self.reg_mode}_epsilon_{self.eps}.png'
            	    name_fig = os.path.join(dir_to_save, name_fig)
            	    
            	    self.plot_2d_mapping_discrete_nu(nu_data_val = nu_data,
                        	   		     mu_sampler = mu_sampler, 
                        	                     random_state = random_state_plot, 
                        	                     plot_mapping_title = plot_mapping_title,
                                                     save_plot = True,
                                                     name_fig = name_fig)     

            loss_arr_batch.append(loss_batch)
            loss_arr_val.append(loss_val)
     
        if make_gif:
            cmdline = 'convert -delay 100 '
            output = f'gaussians_optimizer_{optimizer_mode}_lr_{lr}_regularization_type_{self.reg_mode}_epsilon_{self.eps}.gif'
            for epoch in range(0, epochs, epoch_step_to_print):
                name_fig = f'gaussians_optimizer_{optimizer_mode}_lr_{lr}_epoch_{epoch + 1}_regularization_type_{self.reg_mode}_epsilon_{self.eps}.png'
                name_fig = os.path.join(dir_to_save, name_fig)
                cmdline += name_fig + ' '
            cmdline += output
            os.system(cmdline)
            
    def optimal_map_learning_algo_2(self, epochs = epochs_default, batch_size = batch_size_default,
                                  random_state_val = random_state_default,
                                  random_states_train = random_states_train_default,
                                  mu_sampler = mu_sampler_default, 
                                  index_sampler = index_sampler,
                                  nu_data = nu_data_val_default,
                                  lr = lr_default,
                                  loss_arr_batch = [],
                                  loss_arr_val = [],
                                  epoch_step_to_print = epoch_step_to_print_default):
        
        if (self.v.shape[0] != nu_data.shape[0]):
            raise ValueError("Vector v and nu_data should be the same size!")
        
        for epoch in range(epochs):

            x_batch = mu_sampler(random_state = random_states_train[epoch], batch_size = batch_size)
            
            indexes_to_choice = index_sampler(nu_data_shape = nu_data.shape[0], 
                                              batch_size = batch_size, 
                                              random_state = random_states_train[epoch], 
                                              device = self.device)
            y_batch = nu_data[indexes_to_choice, :]
            u_batch = (self.u)(x_batch)
            v_batch = (self.v)[indexes_to_choice]
            
            start_time = time.time()
            
            self.f_net.zero_grad()
            #self.f_net.train()
            map_batch = (self.f_net)(x_batch)
            loss_batch = self.mapping_OT_loss_estimation(u_batch, v_batch, x_batch, y_batch, map_batch)
            
            loss_batch.backward()
            #data_nu = data_nu.to(device)
            #data_mu = data_mu.to(device)

            f_params_dict = {params_name: params for params_name, params in zip(self.f_net.state_dict(), 
                                                                             self.f_net.parameters())}
            
            f_grad_dict = {params_name: params.grad*lr
                               for params_name, params in zip(self.f_net.state_dict(), self.f_net.parameters())}
            
            for params_name, params in self.f_net.state_dict().items():
                self.f_net.state_dict()[params_name].data.copy_(f_params_dict[params_name] - \
                                                           f_grad_dict[params_name])

            end_time = time.time()
            consumed_time = end_time - start_time

            loss_batch = loss_batch.item()
            
            x_batch_val = mu_sampler(random_state = random_state_val, batch_size = nu_data.shape[0])
            y_batch_val = nu_data
            
            u_batch_val = (self.u)(x_batch_val)
            v_batch_val = self.v
            
            self.f_net.eval()
            map_batch_val = (self.f_net)(x_batch_val)
            
            loss_val = self.mapping_OT_loss_estimation(u_batch_val, v_batch_val, 
                                                       x_batch_val, y_batch_val, map_batch_val)
            
            loss_val = loss_val.item()
            
            if (epoch % epoch_step_to_print == 0):
            	print("------------------------------")
            	print(f"Epoch_num = {epoch + 1}")
            	print(f"Consumed time = {consumed_time} seconds")
            	print(f"Loss estimation on sampled data = {loss_batch}")
            	print(f"Loss estimation on validation data = {loss_val}")

            loss_arr_batch.append(loss_batch)
            loss_arr_val.append(loss_val)
            
    def plot_2d_mapping_discrete_nu(self, nu_data_val = nu_data_val_default,
                        	   mu_sampler = mu_sampler_default, 
                        	   random_state = random_state_default, 
                        	   plot_mapping_title = plot_mapping_title_default,
                                   save_plot = False,
                                   name_fig = None,
                                   show_plot = False):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.set_xlabel(r'$x$') 
        ax.set_ylabel(r'$y$') 
        plt.title(plot_mapping_title) 


        nu_data_plot = nu_data_val.cpu().detach().numpy()
        mu_data_plot = mu_sampler(random_state = random_state, 
                                 batch_size = nu_data_plot.shape[0])

        self.f_net.eval()
        mapping = self.f_net(mu_data_plot)
        mapping = mapping.cpu().detach().numpy()

        mu_data_plot = mu_data_plot.cpu().detach().numpy()

        ax.scatter(mu_data_plot[:, 0], mu_data_plot[:, 1], 
                    label = r'$\mu$-s gaussian', marker='+')
        ax.scatter(nu_data_plot[:, 0], nu_data_plot[:, 1], 
                    label = r'$\nu$-s gaussians', marker='+', color = 'r')

        ax.scatter(mapping[:, 0], mapping[:, 1], 
                    label = r'result mapping', marker='+', color = 'g')

        #plt.scatter(data_mu_validate_plot[:, 0], data_mu_validate_plot[:, 1], 
        #            label = r'$\mu$-s gaussians', marker='+')

        ax.legend()
        ax.grid(True)
        
        if (save_plot and name_fig is not None):
            fig.savefig(name_fig)    
        
        if show_plot:
            plt.show()
        
        else:
            plt.close()
