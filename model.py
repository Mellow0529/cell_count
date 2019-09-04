import torch
import torch.nn as nn
from unet_mini import Unet
from CBAM import CBAM


class Seperate(nn.Module):
    def __init__(self):
        super(Seperate, self).__init__()
    
        self.unet = Unet()
        
    def forward(self, x):
        x_1 = self.unet(x[:,0,:,:].unsqueeze(dim=1))
        x_2 = self.unet(x[:,1,:,:].unsqueeze(dim=1))
        x_3 = self.unet(x[:,2,:,:].unsqueeze(dim=1))
        x_4 = self.unet(x[:,3,:,:].unsqueeze(dim=1))
        x_5 = self.unet(x[:,4,:,:].unsqueeze(dim=1))
        x_6 = self.unet(x[:,5,:,:].unsqueeze(dim=1))

        out = torch.cat((x_1, x_2, x_3, x_4, x_5, x_6), dim = 1)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.seperate = Seperate()
        self.aggregate = CBAM(6, reduction_ratio=2)
        self.end = nn.Conv2d(6, 1, kernel_size = 1)

    def forward(self, x):
        x = self.seperate(x)
        x = self.aggregate(x)
        x = self.end(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)
    x = torch.randn(1, 6, 128, 128).to(device)
    print('device: {}'.format(device))
    y = net(x)
    print(y.shape)