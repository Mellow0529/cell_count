import os 
import glob
import numpy as np 
from PIL import Image 
from scipy import misc
from skimage import io, transform
import csv
import matplotlib.pyplot as plt


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('New path ' + path + ' created')

def resize_cell(img_path, save_path, shape = 128):
    mkdir(save_path)
    
    paths = sorted(glob.glob(os.path.join(img_path, '*.tif')))
    for path in paths:
        img = Image.open(path)
        img = img.resize((shape, shape))
        img.save(os.path.join(save_path, path.split('\\')[-1]))

    print('Compelete resize')

def resize_label(img_path, save_path, shape = 128):
    mkdir(save_path)
    
    paths = sorted(glob.glob(os.path.join(img_path, '*.tif')))
    for path in paths:
        img = io.imread(path)
        img = transform.rescale(img, 128 / 512) * 16
        io.imsave(os.path.join(save_path, path.split('\\')[-1]), img)

    print('Compelete resize')

def resize_test_cell(img_path, save_path, shape = 128):
    mkdir(save_path)
    
    paths = sorted(glob.glob(os.path.join(img_path, '*cell.tif')))
    for path in paths:
        img = Image.open(path)
        img = img.resize((shape, shape))
        img.save(os.path.join(save_path, path.split('\\')[-1]))

    print('Compelete resize')

def resize_test_label(img_path, save_path, shape = 128):
    mkdir(save_path)
    
    paths = sorted(glob.glob(os.path.join(img_path, '*density_map.tif')))
    for path in paths:
        img = io.imread(path)
        img = transform.rescale(img, 128 / 512) * 16
        io.imsave(os.path.join(save_path, path.split('\\')[-1]), img)

    print('Compelete resize')

def param_calc(img):
    img = np.array(img).astype(np.float32)
    return round(np.mean(img), 4), round(np.std(img), 4)

def img_param(path):
    num = int(len(path) / 6)
    mean = []
    std = []
    mean_tmp = 0.0
    std_tmp = 0.0
    for i in range(6):
        for j in range(num):
            img = Image.open(path[j * 6 + i])
            mean_tmp += param_calc(img)[0]
            std_tmp += param_calc(img)[1]
        mean.append(round(mean_tmp / num, 4))
        std.append(round(std_tmp / num, 4))
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))

def param():
    img_path = sorted(glob.glob(os.path.join('data_crop/train/', '*.tif')))
    img_param(img_path)


if __name__ == '__main__':
    train_img_path = 'data/train/'
    train_label_path = 'data/label/'
    test_path = 'data/test/'

    save_path = 'data_crop/'
    
    # path = sorted(glob.glob(os.path.join(test_path, '*cell.tif')))
    # print(path[0].split('\\')[-1])
    '''
    img = io.imread(path[0])
    print(img.sum())
    img = transform.rescale(img, 128 / 512) * 16
    print(img.shape)
    print(img.sum())
    plt.imshow(img)
    plt.show()
    '''
    # resize_cell(train_img_path, save_path + 'train/')
    # resize_label(train_label_path, save_path + 'label')
    # resize_test_cell(test_path, save_path + 'test/')
    # resize_test_label(test_path, save_path + 'test/')
    param()