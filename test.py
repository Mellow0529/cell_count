import torch
import torchvision
import numpy as np
import os
from glob import glob
import PIL
from PIL import Image
import random
import matplotlib.pyplot as plt
from load_data import Dataset, TestDataset
from model import Net
from scipy import misc
from skimage import exposure, io
from model import Net

def show_tensor(img): 
    # img shape: [H, W]
    img = img.cpu()
    if img.requires_grad == True:
        img = img.detach()
    img_np = img.numpy().reshape((512, 512))
    # plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.imshow(img_np)
    plt.show()

random_seed = 60
save_path = 'save_model/pred_map/'
load_model_path = 'save_model/epoch150_loss_35.7894.pth.tar'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(random_seed)

test_set = TestDataset()
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
print('test: {}'.format(len(test_set)))

print('==> Building model...')

net = Net()
net.load_state_dict(torch.load(load_model_path))
net.to(device)

net.eval()
print('device: {}'.format(device))
sum_ = 0
for i, data in enumerate(test_loader):
    img, label = data[0], data[1]
    label = label.type(torch.FloatTensor)
    img = img.type(torch.FloatTensor)
    # print('img max: {}'.format(torch.max(img)))
    # print('img min: {}'.format(torch.min(img)))
    img, label = img.to(device), label.to(device)

    output = net(img) / 100
    # print('pred max: {}'.format(torch.max(output)))
    # print('pred min: {}'.format(torch.min(output)))
    '''
    save_image(img.cpu().data[:,0,:,:], os.path.join(save_path, 'test_{}_img_ch0.bmp'.format(i)))
    save_image(img.cpu().data[:,1,:,:], os.path.join(save_path, 'test_{}_img_ch1.bmp'.format(i)))
    save_image(img.cpu().data[:,2,:,:], os.path.join(save_path, 'test_{}_img_ch2.bmp'.format(i)))
    save_image(img.cpu().data[:,3,:,:], os.path.join(save_path, 'test_{}_img_ch3.bmp'.format(i)))
    save_image(img.cpu().data[:,4,:,:], os.path.join(save_path, 'test_{}_img_ch4.bmp'.format(i)))
    save_image(img.cpu().data[:,5,:,:], os.path.join(save_path, 'test_{}_img_ch5.bmp'.format(i)))
    # show_tensor(img.cpu().data[0][1])
    save_image(label.cpu().data, os.path.join(save_path, 'test_{}_label.bmp'.format(i)))
    # show_tensor(label.cpu().data.squeeze(0).squeeze(0))
    '''
    # save_image(output.cpu().data, os.path.join(save_path, 'test_{}_pred.bmp'.format(i)))
    # print('Pred saved.')
    # out_np = output.detach().cpu().numpy().reshape((512, 512))
    
    # io.imsave(os.path.join(save_path, 'test_{}_pred.bmp'.format(i)), out_np)

    # print('label max: {}'.format(torch.max(label)))
    # print('label min: {}'.format(torch.min(label)))

    # print('pred max: {}'.format(torch.max(output)))
    # print('pred min: {}'.format(torch.min(output)))
    if i > 4:
        print('test {}: label = {:.4f}, pred = {:.4f}, abs: {}.'.format(i, float(label.cpu().data.sum()), float(output.cpu().data.sum()), 
            abs(label.cpu().data.numpy().sum() - output.cpu().data.numpy().sum())))
        # print('abs: {}'.format(abs(label.cpu().data.numpy().sum() - output.cpu().data.numpy().sum())))
        sum_ += abs(label.cpu().data.numpy().sum() - output.cpu().data.numpy().sum())

    '''
    fig, ax = plt.subplots()
    plt.subplot(241)
    plt.imshow(img[0][0].cpu().detach().numpy())
    plt.subplot(242)
    plt.imshow(img[0][1].cpu().detach().numpy())
    plt.subplot(243)
    plt.imshow(img[0][2].cpu().detach().numpy())
    plt.subplot(244)
    plt.imshow(img[0][3].cpu().detach().numpy())
    plt.subplot(245)
    plt.imshow(img[0][4].cpu().detach().numpy())
    plt.subplot(246)
    plt.imshow(img[0][5].cpu().detach().numpy())
    plt.subplot(247)
    plt.imshow(output[0][0].cpu().detach().numpy(), cmap='jet')
    plt.subplot(248)
    plt.imshow(label[0][0].cpu().detach().numpy(), cmap='jet')

    # plt.show()
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'test_{}.png'.format(i)), dpi = 300, quality = 95, bbox_inches = 'tight')
    '''
    

print(sum_ / 15)