#import statements
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import trange

#set up GPU
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

class AE(nn.Module):
    def __init__(self, c, debug=False):
        self.debug = debug
        self.n = 1
        super(AE, self).__init__()
        self.losses = []
        self.enc_conv1 = nn.Conv2d( c,  64, 3, 1, 1)
        self.max_pool1 = nn.MaxPool2d(2)
        self.batch_norm1 = nn.BatchNorm2d(64)
        
        self.enc_conv2 = nn.Conv2d( 64, 32, 3, 1, 1)
        self.max_pool2 = nn.MaxPool2d(2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        
        self.enc_conv3 = nn.Conv2d(32, 16, 3, 1, 1)
        self.max_pool3 = nn.MaxPool2d(2, padding=1)
        
        self.dec_conv1 = nn.Conv2d(16, 64, 3, 1, 1)
        self.dec_up1 = nn.Upsample(scale_factor=2)
        
        self.dec_conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.dec_up2 = nn.Upsample(scale_factor=2)
        
        self.dec_conv3 = nn.Conv2d(32, 16, 3, 1)
        self.dec_up3 = nn.Upsample(scale_factor=2)
        
        self.dec_conv4 = nn.Conv2d(16, c, 3, 1, 1)
        
    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.batch_norm1(self.max_pool1(x))
        x = F.relu(self.enc_conv2(x))
        x = self.batch_norm2(self.max_pool2(x))
        x = F.relu(self.enc_conv3(x))
        x = self.max_pool3(x)
        return x
    
    
    
    def decode(self, z):
        z = F.relu(self.dec_conv1(z))
        z = self.dec_up1(z)
        z = F.relu(self.dec_conv2(z))
        z = self.dec_up2(z)
        z = F.relu(self.dec_conv3(z))
        z = self.dec_up3(z)
        return torch.sigmoid(self.dec_conv4(z))

    def forward(self, x):
        return self.decode(self.encode(x))
    
    def learn(self, dl, epochs=10, optimizer=None, loss_fcn=None, tol = 1e-5, progress_bar=True, plot=True):
        epoch_range = trange(epochs) if progress_bar else range(epochs)
        for epoch in epoch_range:
            total_loss = 0.
            for x, t in dl:
                x, t = x.to(device), t.to(device)
                y = self(x)
                loss = loss_fcn(y, t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()*len(t)
            self.losses.append(total_loss/len(dl.dataset))
            if epoch > 0 and abs(self.losses[-1] - self.losses[-2]) < tol:
                print(f'Early Termination (Epoch {epoch})')
                break
        if plot:
            plt.figure(figsize=(6,4))
            plt.plot(self.losses, label='total')
            plt.legend()
            plt.yscale('log')
            plt.show()
        return self.losses[-1]


class VAE(nn.Module):
    def __init__(self, c, debug=False):
        self.debug = debug
        self.n = 1
        super(VAE, self).__init__()
        self.losses = []
        self.enc_conv1 = nn.Conv2d( c,  64, 3, 1, 1)
        self.max_pool1 = nn.MaxPool2d(2)
        self.batch_norm1 = nn.BatchNorm2d(64)
        
        self.enc_conv2 = nn.Conv2d( 64, 32, 3, 1, 1)
        self.max_pool2 = nn.MaxPool2d(2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        
        self.enc_conv3 = nn.Conv2d(32, 16, 3, 1, 1)
        self.max_pool3 = nn.MaxPool2d(2, padding=1)
        
        self.enc_mean = nn.Linear(10000, 1024)
        self.enc_var = nn.Linear(10000, 1024)
        
        self.dec_fc = nn.Linear(1024, 10000)
        
        self.dec_conv1 = nn.Conv2d(16, 64, 3, 1, 1)
        self.dec_up1 = nn.Upsample(scale_factor=2)
        
        self.dec_conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.dec_up2 = nn.Upsample(scale_factor=2)
        
        self.dec_conv3 = nn.Conv2d(32, 16, 3, 1)
        self.dec_up3 = nn.Upsample(scale_factor=2)
        
        self.dec_conv4 = nn.Conv2d(16, c, 3, 1, 1)
        
    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.batch_norm1(self.max_pool1(x))
        x = F.relu(self.enc_conv2(x))
        x = self.batch_norm2(self.max_pool2(x))
        x = F.relu(self.enc_conv3(x))
        x = self.max_pool3(x)
        x = torch.flatten(x, 1)
        mean, log_var = self.enc_mean(x), self.enc_var(x)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return mean + eps * std    
    
    def decode(self, z):
        z = F.relu(self.dec_fc(z))
        z = nn.Unflatten(1, (16, 25, 25))(z)
        z = F.relu(self.dec_conv1(z))
        z = self.dec_up1(z)
        z = F.relu(self.dec_conv2(z))
        z = self.dec_up2(z)
        z = F.relu(self.dec_conv3(z))
        z = self.dec_up3(z)
        return torch.sigmoid(self.dec_conv4(z))

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        rec = self.decode(z)
        return rec, mean, log_var
    
    def learn(self, dl, epochs=10, optimizer=None, loss_fcn=None, tol = 1e-5, progress_bar=True, plot=True):
        epoch_range = trange(epochs) if progress_bar else range(epochs)
        for epoch in epoch_range:
            total_loss = 0.
            for x, t in dl:
                x, t = x.to(device), t.to(device)
                y, mean, log_var = self(x)
                loss = loss_fcn(y, t, mean, log_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()*len(t)
            self.losses.append(total_loss/len(dl.dataset))
            if epoch > 0 and abs(self.losses[-1] - self.losses[-2]) < tol:
                print(f'Early Termination (Epoch {epoch})')
                break
        if plot:
            plt.figure(figsize=(6,4))
            plt.plot(self.losses, label='total')
            plt.legend()
            plt.yscale('log')
            plt.show()
        return self.losses[-1]


class vae_loss(nn.Module):
    def __init__(self, beta=1):
        super(vae_loss, self).__init__()
        self.beta = beta
    
    def forward(self, pred, target, mean, log_var):
        kl = -0.5 * (1 + log_var - mean ** 2 - torch.exp(log_var)).sum()
        mse = F.mse_loss(pred, target)
        return kl * self.beta + mse
