import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
# datasets
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
from torch.optim import Adam


BATCH_SIZE = 128
IMG_SIZE = 64

def load_transformed_dataset():
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x*2)-1)
    ])

    train  = torchvision.datasets.StanfordCars(root='/ssd_scratch/cvit/anirudhkaushik/datasets/stanford-car-dataset-by-classes-folder/car_data/', split="train", download=False, transform=data_transforms)
    test   = torchvision.datasets.StanfordCars(root='/ssd_scratch/cvit/anirudhkaushik/datasets/stanford-car-dataset-by-classes-folder/car_data/', split="test", download=False, transform=data_transforms)


    return torch.utils.data.ConcatDataset([train, test])


data = load_transformed_dataset()
data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

from noise_scheduler import NoiseScheduler
beta_start = 0.0001
beta_end = 0.02
timesteps = 300


noise_scheduler = NoiseScheduler(beta_start, beta_end, timesteps, 32)
noise_scheduler.beta_scheduler()

image = next(iter(data_loader))[0]

# plt.figure(figsize=(10,10))
num_images = 10
stepsize = int(timesteps/num_images)

for idx in range(0, timesteps, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    # plt.subplot(1, num_images+1, int((idx/stepsize)+1))
    # remove axis of subplot
    # plt.axis('off')
    image, noise  = noise_scheduler.forward_diffusion(image, t)
    noise_scheduler.show(image.cpu().detach())

from unet import SimpleUNET

model = SimpleUNET()
print("Num params: ", sum(p.numel() for p in model.parameters()))

def get_loss(model, x_0, t):
    x_noisy, noise = noise_scheduler.forward_diffusion(x_0, t)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

epoch_start = 0
def load_model_from_checkpoint(mode, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Restarting from epoch {epoch}")
    

def restart_last_checkpoint(model):
    PATH = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/unet_latest.pt'
    load_model_from_checkpoint(model, PATH)
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# load_model_from_checkpoint(model, "/ssd_scratch/cvit/anirudhkaushik/checkpoints/unet.pt")
model = SimpleUNET()
model = torch.nn.DataParallel(model)
restart_last_checkpoint(model)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 500

for epoch in range(epoch_start, epochs):
    for step, batch in enumerate(data_loader):
        optimizer.zero_grad()

        t = torch.randint(0, timesteps, (BATCH_SIZE,), device=device).long()

        loss = get_loss(model, batch[0], t)
        loss.backward()
        optimizer.step()

        if epoch %5 == 0 and step == 0:
            print(f"Epoch: {epoch}, step: {step:03d}, Loss: {loss.item()}")
            noise_scheduler.sample_plot_image(IMG_SIZE, model)

            PATH = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/unet_{epoch}_epoch.pt'
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, PATH)
            

            PATH = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/unet_latest.pt'
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, PATH)