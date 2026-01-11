#! /usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

# =============================================================================
# 設定
# =============================================================================
latent_dim = 100              # 潜在ベクトル次元
BATCH_SIZE = 64
NUM_EPOCHS = 50
LR = 0.0002
BETA1 = 0.5

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ディレクトリ作成
os.makedirs('outputs', exist_ok=True)


# =============================================================================
# データローダー
# =============================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# =============================================================================
# 生成ネットワーク（Generator）
# =============================================================================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 7, 7)
        return self.conv(x)


# =============================================================================
# 識別ネットワーク（Discriminator）
# =============================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# =============================================================================
# 重み初期化
# =============================================================================
def weights_init(m):
    name = m.__class__.__name__
    if 'Conv' in name or 'Linear' in name:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif 'BatchNorm' in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


# =============================================================================
# 学習
# =============================================================================
# モデル初期化
G = Generator().to(device)
D = Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)

# 損失関数・Optimizer
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

# 固定ノイズ（可視化用）
fixed_noise = torch.randn(64, latent_dim, device=device)

# 損失記録
d_losses, g_losses = [], []

print(f"学習開始: {NUM_EPOCHS} epochs")

for epoch in range(1, NUM_EPOCHS + 1):
    d_loss_sum, g_loss_sum = 0, 0
    
    for real_imgs, _ in dataloader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)

        # ----- Discriminator -----
        D.zero_grad()
        loss_real = criterion(D(real_imgs), real_label)
        
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(z)
        loss_fake = criterion(D(fake_imgs.detach()), fake_label)
        
        loss_D = loss_real + loss_fake
        loss_D.backward()
        opt_D.step()

        # ----- Generator -----
        G.zero_grad()
        loss_G = criterion(D(fake_imgs), real_label)
        loss_G.backward()
        opt_G.step()

        d_loss_sum += loss_D.item()
        g_loss_sum += loss_G.item()

    # エポック終了
    d_avg = d_loss_sum / len(dataloader)
    g_avg = g_loss_sum / len(dataloader)
    d_losses.append(d_avg)
    g_losses.append(g_avg)
    
    print(f"Epoch [{epoch}/{NUM_EPOCHS}] D: {d_avg:.4f}, G: {g_avg:.4f}")

    # サンプル画像保存
    with torch.no_grad():
        samples = (G(fixed_noise) + 1) / 2
    save_image(samples, f'outputs/epoch_{epoch:03d}.png', nrow=8)


# =============================================================================
# 結果保存
# =============================================================================
# 損失グラフ
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator')
plt.plot(g_losses, label='Generator')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_plot.png', dpi=150)
plt.close()

# モデル保存
torch.save({'G': G.state_dict(), 'D': D.state_dict()}, 'dcgan_model.pth')

print("実行完了 出力: outputs/, loss_plot.png, dcgan_model.pth")
