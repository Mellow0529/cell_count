import os 
import numpy as np 
from glob import glob 
from PIL import Image
from skimage import io
import torch 
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor, ToPILImage
import random
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_path = 'data_crop/train/', label_path = 'data_crop/label/'):
        super(Dataset, self).__init__()
        self.channels = 6
        self.size = (512, 512)
        self.input_path = sorted(glob(os.path.join(input_path, '*.tif'))) # only take ch0
        self.label_path = sorted(glob(os.path.join(label_path, '*.tif')))
        
        assert len(self.input_path) == len(self.label_path) * 6

        if not os.path.exists(input_path):
            raise Exception("[!] {} not exists.".format(input_path))
        if not os.path.exists(label_path):
            raise Exception("[!] {} not exists.".format(label_path))
        
        self.tensor_trans = Compose([ToTensor()])
        self.img_transform = Compose([
            # ToTensor(),
            Normalize((0.0553, 0.1099, 0.1638, 0.2174, 0.2706, 0.3235),
                    (0.0179, 0.037, 0.0566, 0.0756, 0.093, 0.1084)),
            ])
        self.label_transform = Compose([
            ToTensor(),        
            ])
        self.length = len(self.label_path)

    def __getitem__(self, index):
        img_1 = Image.open(self.input_path[index * self.channels])
        img_2 = Image.open(self.input_path[index * self.channels + 1])
        img_3 = Image.open(self.input_path[index * self.channels + 2])
        img_4 = Image.open(self.input_path[index * self.channels + 3])
        img_5 = Image.open(self.input_path[index * self.channels + 4])
        img_6 = Image.open(self.input_path[index * self.channels + 5])

        img_1 = self.tensor_trans(img_1)
        img_2 = self.tensor_trans(img_2)
        img_3 = self.tensor_trans(img_3)
        img_4 = self.tensor_trans(img_4)
        img_5 = self.tensor_trans(img_5)
        img_6 = self.tensor_trans(img_6)

        img = torch.cat((img_1, img_2, img_3, img_4, img_5, img_6), 0)
        img = self.img_transform(img)
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        img -= torch.mean(img)

        label = io.imread(self.label_path[index])
        # print('dtype: {}'.format(label.dtype))
        # print('before to_tensor: {}'.format(label.sum()))
        # print('max label: {}'.format(label.max()))
        label = self.label_transform(label) * 100

        return img, label

    def __len__(self):
        return self.length

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, path = 'data_crop/test'):
        super(TestDataset, self).__init__()
        self.channels = 6
        self.img_paths = sorted(glob(os.path.join(path, '*_cell.tif')))
        self.label_paths = sorted(glob(os.path.join(path, '*density_map.tif')))
        
        self.tensor_trans = Compose([ToTensor()])
        self.img_transform = Compose([
            # ToTensor(),
            Normalize((0.0553, 0.1099, 0.1638, 0.2174, 0.2706, 0.3235),
                    (0.0179, 0.037, 0.0566, 0.0756, 0.093, 0.1084)), 
            ])
        assert len(self.img_paths) == len(self.label_paths) * 6
        self.len = int(len(self.label_paths))

    def __getitem__(self, index):
        img_1 = Image.open(self.img_paths[index * self.channels])
        img_2 = Image.open(self.img_paths[index * self.channels + 1])
        img_3 = Image.open(self.img_paths[index * self.channels + 2])
        img_4 = Image.open(self.img_paths[index * self.channels + 3])
        img_5 = Image.open(self.img_paths[index * self.channels + 4])
        img_6 = Image.open(self.img_paths[index * self.channels + 5])

        img_1 = self.tensor_trans(img_1)
        img_2 = self.tensor_trans(img_2)
        img_3 = self.tensor_trans(img_3)
        img_4 = self.tensor_trans(img_4)
        img_5 = self.tensor_trans(img_5)
        img_6 = self.tensor_trans(img_6)

        img = torch.cat((img_1, img_2, img_3, img_4, img_5, img_6), 0)
        img = self.img_transform(img)
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        img -= torch.mean(img)

        label = io.imread(self.label_paths[index])
        # print('dtype: {}'.format(label.dtype))
        # print('before to_tensor: {}'.format(label.sum()))
        # print('max label: {}'.format(label.max()))
        label = self.tensor_trans(label)

        return img, label

    def __len__(self):
        return self.len

if __name__ == '__main__':
    dataset = Dataset()
    test_set = TestDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

    for i, data in enumerate(loader):
        print('img shape: {}'.format(data[0].shape))
        print('img max: {}'.format(torch.max(data[0])))
        print('img min: {}'.format(torch.min(data[0])))
        print('label max: {}'.format(torch.max(data[1])))
        print('label min: {}'.format(torch.min(data[1])))
        print('cell: {}'.format(torch.sum(data[1])))
        '''
        plt.subplot(241)
        plt.imshow(data[0][:,0,:,:].squeeze(0).numpy())
        plt.subplot(242)
        plt.imshow(data[0][:,1,:,:].squeeze(0).numpy())
        plt.subplot(243)
        plt.imshow(data[0][:,2,:,:].squeeze(0).numpy())
        plt.subplot(244)
        plt.imshow(data[0][:,3,:,:].squeeze(0).numpy())
        plt.subplot(245)
        plt.imshow(data[0][:,4,:,:].squeeze(0).numpy())
        plt.subplot(246)
        plt.imshow(data[0][:,5,:,:].squeeze(0).numpy())
        plt.subplot(247)
        plt.imshow(data[1][0].squeeze(0).numpy())
        plt.show()
        '''
        break