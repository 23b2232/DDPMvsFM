# Requirements

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# CONFIGURATION (THE CONTROL PANEL)

MODE = "FLOW"  # Options: "FLOW" or "DIFFUSION"
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 10  # For quick results can be increased to 20 for better quality
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running in {MODE} mode on {DEVICE}")

# THE DATASET (MNIST)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad(2), # Pad 28x28 -> 32x32 for easier U-Net handling
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# THE SHARED U-NET ARCHITECTURE

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (64, 128, 256)
        up_channels = (256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(len(down_channels)):
            in_ch = image_channels if i == 0 else down_channels[i-1]
            out_ch = down_channels[i]
            self.downs.append(Block(in_ch, out_ch, time_emb_dim))

        for i in range(len(up_channels)):
            in_ch = up_channels[i]
            out_ch = up_channels[i+1] if i < len(up_channels)-1 else 64
            self.ups.append(Block(in_ch, out_ch, time_emb_dim, up=True))

        self.output = nn.Conv2d(64, out_dim, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
        return self.output(x)

model = SimpleUNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# TRAINING LOOP 

losses = []

print("Starting training...")
for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(dataloader):
        images = images.to(DEVICE)
        batch_size = images.shape[0]

        # Common: Noise generation
        noise = torch.randn_like(images).to(DEVICE)
        t = torch.rand(batch_size, device=DEVICE) # Uniform t [0,1]

        # BRANCHING LOGIC 
        if MODE == "FLOW":
            # Flow Matching: Linear Interpolation
            # x_t = t * x_1 + (1 - t) * x_0
            t_reshaped = t.view(-1, 1, 1, 1)
            x_t = t_reshaped * images + (1 - t_reshaped) * noise
            target = images - noise # Velocity = Data - Noise

            # Predict velocity
            pred = model(x_t, t)
            loss = loss_fn(pred, target)

        elif MODE == "DIFFUSION":
            # DDPM: Noise Schedule (Simplified Linear)
            # This is a simplified "Variance Preserving" approximation for speed
            t_reshaped = t.view(-1, 1, 1, 1)
            alpha_bar = 1 - t_reshaped # Simple linear schedule proxy

            # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
            x_t = torch.sqrt(alpha_bar) * images + torch.sqrt(1 - alpha_bar) * noise
            target = noise # DDPM predicts the noise

            # Predict noise
            pred = model(x_t, t)
            loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")
            losses.append(loss.item())
import os
os.makedirs("models", exist_ok=True)

# Save Model
torch.save(model.state_dict(), f"models/model_{MODE.lower()}.pth")
print("Training Complete")
