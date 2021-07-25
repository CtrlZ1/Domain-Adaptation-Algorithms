import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt 
import time
import copy
from IPython.display import Image
import os


from .constants import (f_net_default, reg_mode_default, eps_default,
			c_cost_type_default, d_cost_type_default,
			reg_modes_arr,
			dtype_default, device_default, mu_sampler_default, 
			random_state_default)

class Neural_OT:

    def __init__(self, f_net = f_net_default, reg_mode = reg_mode_default, eps = eps_default, 
                 c_cost_type = c_cost_type_default,
                 d_cost_type = d_cost_type_default,
                 dtype = dtype_default, device = device_default):

        if (c_cost_type != 'l2'):
            raise ValueError(f"Only {c_cost_type_default} c cost type is now available :(")
            
        if (d_cost_type != 'l2'):
            raise ValueError(f"Only {d_cost_type_default} d cost type is now available :(")
            
        if (reg_mode not in reg_modes_arr):
            raise ValueError(f"Only {reg_modes_arr[0]} and {reg_modes_arr[1]}" + \
                                " regularization types are now available :(")
        
        f_copy = copy.deepcopy(f_net)
        self.f_net = f_net.to(device)
        self.reg_mode = reg_mode
        self.eps = eps
        self.c_cost_type = c_cost_type
        self.d_cost_type = d_cost_type
        self.dtype = dtype
        self.device = device
        
    def replace_f(self, f_net):
        f_copy = copy.deepcopy(f_net)
        self.f_net = f_copy.to(self.device)
        
    def l2_dist_batch(self, x_batch, y_batch):
        diff = x_batch - y_batch
        
        if len(x_batch.shape) > 1:
            return torch.mul(diff, diff).sum(dim = 1).reshape(-1, 1)   
        
        else:
            return torch.mul(diff, diff).sum()
    
    def H_eps_batch(self, u_batch, v_batch, x_batch, y_batch):
        H_eps = torch.zeros((u_batch.shape[0], 1), dtype=self.dtype, device=self.device)
        
        if (self.c_cost_type == 'l2'):
            
            c_batch = self.l2_dist_batch(x_batch, y_batch)
            
            if self.reg_mode == 'l2':
                relu = nn.ReLU()
                value = relu(u_batch.type(torch.double) + v_batch.type(torch.double) - c_batch.type(torch.double))
                H_eps = value/(2*self.eps)

            if self.reg_mode == 'entropy':
                value = (u_batch + v_batch - c_batch)/self.eps
                H_eps = torch.exp(value)
            
        return H_eps
    
    def F_eps_batch(self, u_batch, v_batch, x_batch, y_batch):
        F_eps = torch.zeros((u_batch.shape[0], 1), dtype = self.dtype, device = self.device)
        
        if (self.c_cost_type == 'l2'):
            
            c_batch = self.l2_dist_batch(x_batch, y_batch)
            
            if self.reg_mode == 'l2':
                relu = nn.ReLU()

                value = relu(u_batch.type(torch.double) + v_batch.type(torch.double) - c_batch.type(torch.double))
                F_eps = -(value ** 2)/(4.0*self.eps)

            if self.reg_mode == 'entropy':
                value = (u_batch + v_batch - c_batch)/self.eps
                F_eps = -self.eps*torch.exp(value)
            
        return F_eps

        
    def dual_OT_loss_estimation(self, u_batch, v_batch, x_batch, y_batch):
        
        num_estimators = x_batch.shape[0]
        
        loss_u_part = torch.sum(u_batch)
        loss_v_part = torch.sum(v_batch)
        
        F_eps = self.F_eps_batch(u_batch, v_batch, x_batch, y_batch)
        
        loss = torch.mean(u_batch.type(torch.double) + v_batch.type(torch.double) + F_eps.type(torch.double))
        
        return -loss
        
    def mapping_OT_loss_estimation(self, u_batch, v_batch, x_batch, y_batch, map_batch):

        num_estimators = x_batch.shape[0]
        сur_loss = torch.zeros(1, dtype=self.dtype, device=self.device)
        
        if (self.d_cost_type == 'l2'):
        
            d_batch = self.l2_dist_batch(y_batch, map_batch)
            H_eps = self.H_eps_batch(u_batch, v_batch, x_batch, y_batch)
            сur_loss = torch.mul(d_batch.type(torch.double), H_eps.type(torch.double)).mean()

        return сur_loss
    
    def plot_loss_graphs(self, loss_arr_batch, loss_arr_val, optimizer_mode, lr, plot_mode):
        if (plot_mode == 'dual'):
            first_part_title = r'Training dual parameters $(u, v), $'
            
        if (plot_mode == 'mapping training'):
            first_part_title = r'Training mapping $f$, '
            
        fig = plt.figure(figsize=(10,5))

        plt.xlabel(r'$iter$') 
        plt.ylabel(r'$loss \; estimation$') 
        plt.title(first_part_title + \
                  fr'optimizer = {optimizer_mode}, $lr = {lr}$, regularization type = {self.reg_mode}, $\varepsilon = {self.eps}$') 

        plt.plot(range(len(loss_arr_batch)), loss_arr_batch, label = r'last batch')
        plt.plot(range(len(loss_arr_batch)), loss_arr_val, label = r'validation data')

        plt.legend()
        plt.grid(True) 
        
    def create_path_to_gif(self, optimizer_mode, lr):
        gif = f'gaussians_optimizer_{optimizer_mode}_lr_{lr}_regularization_type_{self.reg_mode}_epsilon_{self.eps}.gif'
        return os.path.expanduser(gif)
