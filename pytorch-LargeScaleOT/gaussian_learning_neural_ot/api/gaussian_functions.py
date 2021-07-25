import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt 

def gaussian_sampler_2d(gaussian_center, cov_matrix):
    mu_distr = MultivariateNormal(gaussian_center, cov_matrix)
    return mu_distr

def gaussian_data_sampling(gaussian_center, cov_matrix, data_num, device = None):
    sampler = gaussian_sampler_2d(gaussian_center, cov_matrix)
    data = sampler.sample(sample_shape=torch.Size([data_num]))
    if (device is not None):
        data = data.to(device)

    return data
    
def plot_data_gaussians(data_mu, data_nu):
    fig = plt.figure(figsize=(10,10))

    plt.xlabel(r'$x$') 
    plt.ylabel(r'$y$') 
    plt.title('1 and 8 gaussians') 

    plt.scatter(data_mu[:, 0], data_mu[:, 1], label = r'$\mu$-s gaussian', marker='+')
    plt.scatter(data_nu[:, 0], data_nu[:, 1], label = r'$\nu$-s gaussians', marker='+', color = 'r')

    plt.legend()
    plt.grid(True) 

def gaussian_data_sampling_nu(centers_nu, cov_matrix, batch_size, random_state, device = None):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    index_to_choice = np.random.randint(centers_nu.shape[0], size = batch_size)
    data_nu = gaussian_data_sampling(centers_nu[index_to_choice[0]], cov_matrix, 1)

    for i in range(1, batch_size):
        cur_data_nu = gaussian_data_sampling(centers_nu[index_to_choice[i]], cov_matrix, 1)
        data_nu = torch.cat((data_nu, cur_data_nu), 0)
    
    if (device is not None):
        data_nu = data_nu.to(device)

    return data_nu

def gaussian_data_sampling_mu(center_mu, cov_matrix, batch_size, random_state, device = None):
    torch.manual_seed(random_state)
    return gaussian_data_sampling(center_mu, cov_matrix, batch_size, device)

def index_sampler(nu_data_shape, batch_size, random_state, device = None):
    np.random.seed(random_state)
    indexes_to_choice = np.random.randint(nu_data_shape, size = batch_size)
    indexes_to_choice = torch.from_numpy(indexes_to_choice)
    
    if (device is not None):
        indexes_to_choice = indexes_to_choice.to(device)
        
    return indexes_to_choice.long()

def nu_sampler_from_discrete_distr(nu_data, batch_size, random_state, device = None):
    indexes_to_choice = index_sampler(nu_data_shape = nu_data.shape[0], 
                                      batch_size = batch_size, 
                                      random_state = random_state, 
                                      device = device)
    
    return nu_data[indexes_to_choice, :]
    
    
