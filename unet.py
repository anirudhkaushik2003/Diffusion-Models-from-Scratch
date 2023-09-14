import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim//2
        embeddings = math.log(10000)/ (half_dim -1)
        embeddings = torch.exp(torch.arange(half_dim, device=device)* -embeddings)
        embeddings = time[:,None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_embedding_dim, up=False):
        super(Block, self).__init__()
        self.time_mlp = nn.Linear(time_embedding_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4,2,1)

        else: 
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4,2,1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.bnorm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()


    def forward(self, x, t):
        # first conv
        h = self.bnorm(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        # extend last 2 dimensions

        time_emb = time_emb[(..., )+ (None, )*2]

        # Add time channel
        h = h + time_emb

        # second conv
        h = self.bnorm(self.relu(self.conv2(h)))

        # Down or up sample
        return self.transform(h)

class SimpleUNET(nn.Module):

    def __init__(self):
        super(SimpleUNET, self).__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1
        time_embedding_dim = 32

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_embedding_dim) for i in range(len(down_channels)-1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_embedding_dim, up=True) for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    
    def forward(self, x, timestep):
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x,t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim = 1)
            x = up(x,t)

        return self.output(x)
    

