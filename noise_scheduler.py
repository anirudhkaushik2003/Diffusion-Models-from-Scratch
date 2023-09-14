import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class NoiseScheduler():
    def __init__(self, beta_start, beta_end, timesteps, batch_size):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def beta_scheduler(self):
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.betas = self.betas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)

        self.posterior_variance = self.betas*(1-self.alphas_cumprod_prev)/(1.0 - self.alphas_cumprod)
    
    def get_index_at_t(self, t, vals, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,)*(len(x_shape) -1))).to(t.device)
        
    def forward_diffusion(self, x_0, t):

        x_0 = x_0.to(self.device)
        noise = torch.randn_like(x_0).to(self.device)
        self.sqrt_alphas_cumprod_t = self.get_index_at_t(t, self.sqrt_alphas_cumprod, x_0.shape).to(self.device)
        self.sqrt_one_minus_alphas_cumprod_t = self.get_index_at_t(t, self.sqrt_one_minus_alphas_cumprod, x_0.shape).to(self.device)

        return self.sqrt_alphas_cumprod_t*x_0 + self.sqrt_one_minus_alphas_cumprod_t*noise, noise
