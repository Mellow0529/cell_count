import torch
import torch.nn as nn
import math

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class Unet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        # self.dconv_down3 = double_conv(128, 256)
        # self.dconv_down4 = double_conv(256, 512)
        self.bottom = double_conv(128, 256)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        # self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        # print('conv1: {}'.format(conv1.shape))
        x = self.maxpool(conv1)
        # print('conv1 -> maxpool: {}'.format(x.shape))

        conv2 = self.dconv_down2(x)
        # print('conv2: {}'.format(conv2.shape))
        x = self.maxpool(conv2)
        # print('conv2 -> maxpool: {}'.format(x.shape))

        x = self.bottom(x)
        # print('bottom: {}'.format(x.shape))

        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

if __name__ == '__main__':
    net = Unet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    x = torch.randn(1, 1, 512, 512).to(device)
    print('device: {}'.format(device))
    y = net(x)
    print(y.shape)