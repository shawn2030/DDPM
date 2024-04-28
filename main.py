import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from unet import UNet
from linear_noise_scheduler import LinearNoiseScheduler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 15
BATCH_SIZE = 128
LR_RATE = 0.0001

NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02


def train_ddpm(train_loader, scheduler, ddpm_model, optimizer, criterion):

    print()
    print("Diffusion Model Training Starts ...")
    steps = 0
    losses = []

    for epoch in tqdm(range(NUM_EPOCHS), desc = "EPOCHS"):
            
        for batch_idx, (x, label) in enumerate(tqdm(train_loader, desc = "Batches", leave = False)):
            steps += 1
            optimizer.zero_grad()

            x = x.to(DEVICE)

            # sample random noise
            noise = torch.randn_like(x).to(DEVICE)

            # sample timestep t
            t = torch.randint(0, NUM_TIMESTEPS, (x.shape[0],)).to(DEVICE)

            # add noise to images according to the time step
            noisy_im = scheduler.add_noise(x, noise, t)
            noise_pred = ddpm_model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item()) 
            loss.backward()
            optimizer.step()

    print()
    print("Diffusion Model Training Ends ...")


    plt.figure()
    plt.title('DDPM Loss Graph')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.plot(range(steps), losses, color = 'red')
    plt.savefig('plots/Loss.png')
    plt.close()

    return ddpm_model


def test_ddpm(ddpm_model, scheduler: LinearNoiseScheduler):
    ddpm_model.eval()
    with torch.no_grad():
        xt = torch.randn((100, 1, 28, 28)).to(DEVICE)

        for i in tqdm(reversed(range(NUM_TIMESTEPS))):
            noise_pred = ddpm_model(xt, torch.as_tensor(i).unsqueeze(0).to(DEVICE))

            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(DEVICE))

            ims = torch.clamp(xt, -1, 1).detach().cpu()
            ims = ( ims + 1 ) / 2
            grid = make_grid(ims, nrow= 10)
            img = torchvision.transforms.ToPILImage()(grid)
            img.save('output/x0_{}.png'.format(i))
            img.close()

            
def main():

    # LOAD MNIST DATASET
    transform = transforms.Compose([
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
            ])
    
    dataset = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)

    train_loader = DataLoader(dataset=dataset, batch_size= BATCH_SIZE, shuffle=True)

    # Create an object of noise scheduler
    scheduler = LinearNoiseScheduler(num_time_steps=NUM_TIMESTEPS, beta_start=BETA_START, beta_end=BETA_END, device=DEVICE)

    # Create Object of Model
    ddpm_model = UNet(im_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(ddpm_model.parameters(), lr=LR_RATE, weight_decay=1e-4)
    loss_criterion = nn.MSELoss()

    ddpm_model = train_ddpm(train_loader=train_loader, scheduler=scheduler, ddpm_model=ddpm_model, optimizer=optimizer, criterion= loss_criterion)
    test_ddpm(ddpm_model, scheduler)

if __name__ == "__main__":
    main()



    
