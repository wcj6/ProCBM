import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import pdb 
from PIL import Image


class SkinDataset(Dataset):
    # 8:1:1
    def __init__(self, dataset_dir, mode='train', transforms=None, flag=0, debug=False, config=None, return_concept_label=False):
        self.mode = mode
        if debug:
            dataset_dir = dataset_dir + 'debug'
        print(dataset_dir)
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.flag = flag
        self.args = config
        self.return_concept_label = return_concept_label
        
       
        print('start loading %s dataset'%mode)
        
        data = np.load(dataset_dir+'dataList.npy', mmap_mode='r', allow_pickle=False)
        label = np.load(dataset_dir+'labelList.npy', mmap_mode='r', allow_pickle=False)
        
        rng = np.random.default_rng(29)
        shuffled_indices = rng.permutation(data.shape[0])
        
        self.dataList = data
        self.labelList = label

        shuffled_label = label[shuffled_indices]
        
        self.origin_size = [data.shape[1], data.shape[2]]


        
        index_list = np.zeros((0), dtype=np.int64)
        for i in range(7):
            print('load class %d'%i)
            num = (shuffled_label==i).sum()
            num = num // 5
            
            index = np.where(shuffled_label == i)[0]
            test_index = index[num*flag:num*(flag+1)]
            train_index = np.array(list(set(index) - set(test_index)))

            num_val = len(test_index) // 2
            if self.mode == 'train':
                index_list = np.concatenate((index_list, shuffled_indices[train_index]), axis=0)


            elif self.mode == 'val':
                # index_list = np.arange(data.shape[0])
                index_list = np.concatenate((index_list, shuffled_indices[test_index[:num_val]]), axis=0)

                # train
                # index_list = np.concatenate((index_list, shuffled_indices[test_index[:num_val]]), axis=0)

            else:
                index_list = np.concatenate((index_list, shuffled_indices[test_index[num_val:]]), axis=0)

        self.index_list = index_list
        
        
        self.concept_label_map = [
            [0, 0, 0, 0, 0, 0, 0], # MEL
            [1, 1, 1, 1, 1, 1, 0], # NV
            [2, 0, 2, 2, 2, 0, 1], # BCC
            [3, 0, 0, 3, 3, 0, 2], # AKIEC
            [4, 2, 1, 4, 4, 1, 3], # BKL
            [5, 1, 1, 5, 5, 1, 0], # DF
            [6, 3, 1, 6, 1, 2, 0], # VASC
        ]

        print('load done')
    
    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        real_idx = self.index_list[idx]
        img = self.dataList[real_idx]
        


        label = int(self.labelList[real_idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.return_concept_label:
            return img, label, np.array(self.concept_label_map[label])
        else:
            return img, label
   
    def random_deform(self, img):
        
        img = img.unsqueeze(0)

        dx = (torch.randn_like(img[0, 0, :, :])*2-1) * self.args.deform_scale
        dy = (torch.randn_like(img[0, 0, :, :])*2-1)  * self.args.deform_scale

        dx = self.smooth(dx.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()
        dy = self.smooth(dy.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()


        x_line = torch.linspace(-1, 1, steps=img.shape[2]).unsqueeze(0)
        x_line = x_line.repeat(img.shape[2], 1)
        y_line = torch.linspace(-1, 1, steps=img.shape[3]).unsqueeze(1)
        y_line = y_line.repeat(1, img.shape[3])

        x_line += dx
        y_line += dy

        grid = torch.stack((x_line, y_line), dim=2)
        grid = grid.unsqueeze(0)

        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
    
        img = img.squeeze(0)

        return img


    def random_affine(self, img):
        
        img = img.unsqueeze(0)

        scale_x = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        scale_y = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        
        shear_x = np.random.random() * self.args.scale
        shear_y = np.random.random() * self.args.scale
        #shear_x = 0
        #shear_y = 0


        angle = np.random.randint(-self.args.angle, self.args.angle)
        angle = (angle / 180.) * math.pi

        theta_scale = torch.tensor([[scale_x, shear_x, 0],
                                    [shear_y, scale_y, 0],
                                    [0, 0, 1]]).float()
        theta_rotate = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                    [math.sin(angle), math.cos(angle), 0],
                                    [0, 0, 1]]).float()

        theta = torch.mm(theta_scale, theta_rotate)[0:2, :]
        grid = F.affine_grid(theta.unsqueeze(0), img.size(), align_corners=True)
        
        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        img = img.squeeze(0)

        return img

class busiDataset(Dataset):
    # 8:1:1
    def __init__(self, dataset_dir, mode='train', transforms=None, flag=0, debug=False, config=None, return_concept_label=False):
        self.mode = mode
        if debug:
            dataset_dir = dataset_dir + 'debug'
        print(dataset_dir)
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.flag = flag
        self.args = config
        self.return_concept_label = return_concept_label

        print('start loading %s dataset' % mode)

        # 加载数据和标签
        data = np.load(dataset_dir + 'dataList.npy', mmap_mode='r', allow_pickle=False)
        label = np.load(dataset_dir + 'labelList.npy', mmap_mode='r', allow_pickle=False)

        # 数据乱序
        rng = np.random.default_rng(29)
        shuffled_indices = rng.permutation(data.shape[0])

        self.dataList = data
        self.labelList = label

        shuffled_label = label[shuffled_indices]

        self.origin_size = [data.shape[1], data.shape[2]]

        self.label_map = {
            0: 0, 1: 0, 5: 0,  # Benign
            2: 1, 3: 1, 6: 1,  # Malignant
            4: 2                # Other
        }
        
       
        # 数据划分
        index_list = np.zeros((0), dtype=np.int64)
        for i in range(3):  # 遍历三分类的类别
            print(f'load class {i}')
            num = (shuffled_label == i).sum()
            num = num // 5

            index = np.where(shuffled_label == i)[0]
            test_index = index[num * flag:num * (flag + 1)]
            train_index = np.array(list(set(index) - set(test_index)))

            num_val = len(test_index) // 2
            if self.mode == 'train':
                index_list = np.concatenate((index_list, shuffled_indices[train_index]), axis=0)

            elif self.mode == 'val':
                index_list = np.concatenate((index_list, shuffled_indices[test_index[:num_val]]), axis=0)
            else:
                index_list = np.concatenate((index_list, shuffled_indices[test_index[num_val:]]), axis=0)

        self.index_list = index_list

        # 定义三分类的概念标签映射（如果需要）
        self.concept_label_map = [
            [0, 0, 0,0,0,0],  # Benign
            [1, 1, 1,1, 1, 1],  # Malignant
            [2, 2, 2,2, 2, 2],  #
        ]

        print('load done')

    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        real_idx = self.index_list[idx]
        img = self.dataList[real_idx]
        label = int(self.labelList[real_idx])

        if self.transforms is not None:
            img = self.transforms(img)

        if self.return_concept_label:
            return img, label, np.array(self.concept_label_map[label])
        else:
            return img, label

    def random_deform(self, img):
        img = img.unsqueeze(0)

        dx = (torch.randn_like(img[0, 0, :, :]) * 2 - 1) * self.args.deform_scale
        dy = (torch.randn_like(img[0, 0, :, :]) * 2 - 1) * self.args.deform_scale

        dx = self.smooth(dx.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()
        dy = self.smooth(dy.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()

        x_line = torch.linspace(-1, 1, steps=img.shape[2]).unsqueeze(0)
        x_line = x_line.repeat(img.shape[2], 1)
        y_line = torch.linspace(-1, 1, steps=img.shape[3]).unsqueeze(1)
        y_line = y_line.repeat(1, img.shape[3])

        x_line += dx
        y_line += dy

        grid = torch.stack((x_line, y_line), dim=2)
        grid = grid.unsqueeze(0)

        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        img = img.squeeze(0)

        return img

    def random_affine(self, img):
        img = img.unsqueeze(0)

        scale_x = np.random.random() * (2 * self.args.scale) + 1 - self.args.scale
        scale_y = np.random.random() * (2 * self.args.scale) + 1 - self.args.scale

        shear_x = np.random.random() * self.args.scale
        shear_y = np.random.random() * self.args.scale

        angle = np.random.randint(-self.args.angle, self.args.angle)
        angle = (angle / 180.) * math.pi

        theta_scale = torch.tensor([[scale_x, shear_x, 0],
                                    [shear_y, scale_y, 0],
                                    [0, 0, 1]]).float()
        theta_rotate = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                     [math.sin(angle), math.cos(angle), 0],
                                     [0, 0, 1]]).float()

        theta = torch.mm(theta_scale, theta_rotate)[0:2, :]
        grid = F.affine_grid(theta.unsqueeze(0), img.size(), align_corners=True)

        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        img = img.squeeze(0)

        return img

class idridDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', transforms=None, flag=0, debug=False, config=None,return_concept_label=False):
        self.mode = mode
        if debug:
            dataset_dir = dataset_dir + 'debug'
        print(dataset_dir)
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.flag = flag
        self.args = config
        self.return_concept_label = return_concept_label

        print('Start loading %s dataset' % mode)

        # 加载数据和标签
        data = np.load(dataset_dir + 'dataList.npy', mmap_mode='r', allow_pickle=False)
        label = np.load(dataset_dir + 'labelList.npy', mmap_mode='r', allow_pickle=False)

        # 数据乱序
        rng = np.random.default_rng(29)
        shuffled_indices = rng.permutation(data.shape[0])

        self.dataList = data
        self.labelList = label

        # 假设标签是 0~4，对应五个类别，无需额外映射
        shuffled_label = self.labelList[shuffled_indices]

        self.origin_size = [data.shape[1], data.shape[2]]

        # 数据划分
        index_list = np.zeros((0), dtype=np.int64)
        for i in range(5):  
            print(f'Load class {i}')
            num = (shuffled_label == i).sum()
            num = num // 5

            index = np.where(shuffled_label == i)[0]
            test_index = index[num * flag:num * (flag + 1)]
            train_index = np.array(list(set(index) - set(test_index)))

            num_val = len(test_index) // 2
            if self.mode == 'train':
                index_list = np.concatenate((index_list, shuffled_indices[train_index]), axis=0)

            elif self.mode == 'val':
                index_list = np.concatenate((index_list, shuffled_indices[test_index[:num_val]]), axis=0)
            else:
                index_list = np.concatenate((index_list, shuffled_indices[test_index[num_val:]]), axis=0)

        self.index_list = index_list
          
        self.concept_label_map = [
            [0, 0, 0, 0, 0, 0], # 
            [1, 1, 1, 1, 1, 1], # 
            [2, 2, 2, 2, 2, 2], # 
            [3, 3, 3, 3, 3, 3], # 
            [4, 4, 4, 4, 4, 4], # 
        ]


        print('Loading done')

    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        real_idx = self.index_list[idx]
        img = self.dataList[real_idx]
        label = int(self.labelList[real_idx])

        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.return_concept_label:
            return img, label, np.array(self.concept_label_map[label])
        else:
            return img, label

        # return img, label

class miniddsmDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', transforms=None, flag=0, debug=False, config=None, return_concept_label=False):
        self.mode = mode
        if debug:
            dataset_dir = dataset_dir + 'debug'
        print(dataset_dir)
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.flag = flag
        self.args = config
        self.return_concept_label = return_concept_label
        
       
        print('start loading %s dataset'%mode)
        
        data = np.load(dataset_dir+'dataList.npy', mmap_mode='r', allow_pickle=False)
        label = np.load(dataset_dir+'labelList.npy', mmap_mode='r', allow_pickle=False)
        
        rng = np.random.default_rng(29)
        shuffled_indices = rng.permutation(data.shape[0])
        
        self.dataList = data
        self.labelList = label

        shuffled_label = label[shuffled_indices]
        
        self.origin_size = [data.shape[1], data.shape[2]]
        
        index_list = np.zeros((0), dtype=np.int64)
        for i in range(2):
            print('load class %d'%i)
            num = (shuffled_label==i).sum()
            num = num // 5
            
            index = np.where(shuffled_label == i)[0]
            test_index = index[num*flag:num*(flag+1)]
            train_index = np.array(list(set(index) - set(test_index)))

            num_val = len(test_index) // 2
            if self.mode == 'train':
                index_list = np.concatenate((index_list, shuffled_indices[train_index]), axis=0)

            elif self.mode == 'val':
                index_list = np.concatenate((index_list, shuffled_indices[test_index[:num_val]]), axis=0)
            else:
                index_list = np.concatenate((index_list, shuffled_indices[test_index[num_val:]]), axis=0)

        self.index_list = index_list
        
        
        self.concept_label_map = [
            [0, 0, 0, 0, 0, 0, 0], # benign
            [1, 1, 1, 1, 1, 1, 1], # malignant
        ]

        print('load done')
    
    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        real_idx = self.index_list[idx]
        img = self.dataList[real_idx]
        label = int(self.labelList[real_idx])
        
        # 确保图像数据类型正确
        if isinstance(img, np.ndarray) and img.dtype == np.float32:
            # 如果图像值范围是 [0, 1]
            if img.min() >= 0 and img.max() <= 1:
                img = (img * 255).astype(np.uint8)  # 转换为 [0, 255] 范围内的 uint8 类型
            else:
                # 如果图像值范围不是 [0, 1]，则先进行归一化再转换
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.return_concept_label:
            return img, label, np.array(self.concept_label_map[label])
        else:
            return img, label
   
    def random_deform(self, img):
        
        img = img.unsqueeze(0)

        dx = (torch.randn_like(img[0, 0, :, :])*2-1) * self.args.deform_scale
        dy = (torch.randn_like(img[0, 0, :, :])*2-1)  * self.args.deform_scale

        dx = self.smooth(dx.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()
        dy = self.smooth(dy.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()


        x_line = torch.linspace(-1, 1, steps=img.shape[2]).unsqueeze(0)
        x_line = x_line.repeat(img.shape[2], 1)
        y_line = torch.linspace(-1, 1, steps=img.shape[3]).unsqueeze(1)
        y_line = y_line.repeat(1, img.shape[3])

        x_line += dx
        y_line += dy

        grid = torch.stack((x_line, y_line), dim=2)
        grid = grid.unsqueeze(0)

        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
    
        img = img.squeeze(0)

        return img


    def random_affine(self, img):
        
        img = img.unsqueeze(0)

        scale_x = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        scale_y = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        
        shear_x = np.random.random() * self.args.scale
        shear_y = np.random.random() * self.args.scale
        #shear_x = 0
        #shear_y = 0


        angle = np.random.randint(-self.args.angle, self.args.angle)
        angle = (angle / 180.) * math.pi

        theta_scale = torch.tensor([[scale_x, shear_x, 0],
                                    [shear_y, scale_y, 0],
                                    [0, 0, 1]]).float()
        theta_rotate = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                    [math.sin(angle), math.cos(angle), 0],
                                    [0, 0, 1]]).float()

        theta = torch.mm(theta_scale, theta_rotate)[0:2, :]
        grid = F.affine_grid(theta.unsqueeze(0), img.size(), align_corners=True)
        
        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        img = img.squeeze(0)

        return img

class cmDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', transforms=None, flag=0, debug=False, config=None, return_concept_label=False):
        self.mode = mode
        if debug:
            dataset_dir = dataset_dir + 'debug'
        print(dataset_dir)
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.flag = flag
        self.args = config
        self.return_concept_label = return_concept_label
        
       
        print('start loading %s dataset'%mode)
        
        data = np.load(dataset_dir+'dataList.npy', mmap_mode='r', allow_pickle=False)
        label = np.load(dataset_dir+'labelList.npy', mmap_mode='r', allow_pickle=False)
        
        rng = np.random.default_rng(29)
        shuffled_indices = rng.permutation(data.shape[0])
        
        self.dataList = data
        self.labelList = label

        shuffled_label = label[shuffled_indices]
        
        self.origin_size = [data.shape[1], data.shape[2]]
        
        index_list = np.zeros((0), dtype=np.int64)
        for i in range(2):
            print('load class %d'%i)
            num = (shuffled_label==i).sum()
            num = num // 5
            
            index = np.where(shuffled_label == i)[0]
            test_index = index[num*flag:num*(flag+1)]
            train_index = np.array(list(set(index) - set(test_index)))

            num_val = len(test_index) // 2
            if self.mode == 'train':
                index_list = np.concatenate((index_list, shuffled_indices[train_index]), axis=0)

            elif self.mode == 'val':
                index_list = np.concatenate((index_list, shuffled_indices[test_index[:num_val]]), axis=0)
            else:
                index_list = np.concatenate((index_list, shuffled_indices[test_index[num_val:]]), axis=0)

        self.index_list = index_list
        
        
        self.concept_label_map = [
            [0, 0, 0, 0], # none
            [1, 1, 1, 1], # Cardiomegaly
        ]

        print('load done')
    
    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        real_idx = self.index_list[idx]
        img = self.dataList[real_idx]
        label = int(self.labelList[real_idx])
        
        # 确保图像数据类型正确
        if isinstance(img, np.ndarray) and img.dtype == np.float32:
            # 如果图像值范围是 [0, 1]
            if img.min() >= 0 and img.max() <= 1:
                img = (img * 255).astype(np.uint8)  # 转换为 [0, 255] 范围内的 uint8 类型
            else:
                # 如果图像值范围不是 [0, 1]，则先进行归一化再转换
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.return_concept_label:
            return img, label, np.array(self.concept_label_map[label])
        else:
            return img, label
   
    def random_deform(self, img):
        
        img = img.unsqueeze(0)

        dx = (torch.randn_like(img[0, 0, :, :])*2-1) * self.args.deform_scale
        dy = (torch.randn_like(img[0, 0, :, :])*2-1)  * self.args.deform_scale

        dx = self.smooth(dx.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()
        dy = self.smooth(dy.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()


        x_line = torch.linspace(-1, 1, steps=img.shape[2]).unsqueeze(0)
        x_line = x_line.repeat(img.shape[2], 1)
        y_line = torch.linspace(-1, 1, steps=img.shape[3]).unsqueeze(1)
        y_line = y_line.repeat(1, img.shape[3])

        x_line += dx
        y_line += dy

        grid = torch.stack((x_line, y_line), dim=2)
        grid = grid.unsqueeze(0)

        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
    
        img = img.squeeze(0)

        return img


    def random_affine(self, img):
        
        img = img.unsqueeze(0)

        scale_x = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        scale_y = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        
        shear_x = np.random.random() * self.args.scale
        shear_y = np.random.random() * self.args.scale
        #shear_x = 0
        #shear_y = 0


        angle = np.random.randint(-self.args.angle, self.args.angle)
        angle = (angle / 180.) * math.pi

        theta_scale = torch.tensor([[scale_x, shear_x, 0],
                                    [shear_y, scale_y, 0],
                                    [0, 0, 1]]).float()
        theta_rotate = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                    [math.sin(angle), math.cos(angle), 0],
                                    [0, 0, 1]]).float()

        theta = torch.mm(theta_scale, theta_rotate)[0:2, :]
        grid = F.affine_grid(theta.unsqueeze(0), img.size(), align_corners=True)
        
        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        img = img.squeeze(0)

        return img

class nctDataset(Dataset):
    # 8:2:2
    def __init__(self, dataset_dir, mode='train', transforms=None, flag=0, debug=False, config=None, return_concept_label=False):
        self.mode = mode
        if debug:
            dataset_dir = dataset_dir + 'debug'
        print(dataset_dir)
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.flag = flag
        self.args = config
        self.return_concept_label = return_concept_label
        
       
        print('start loading %s dataset'%mode)
        
        data = np.load(dataset_dir+'dataList.npy', mmap_mode='r', allow_pickle=False)
        label = np.load(dataset_dir+'labelList.npy', mmap_mode='r', allow_pickle=False)
        
        rng = np.random.default_rng(29)
        shuffled_indices = rng.permutation(data.shape[0])
        
        self.dataList = data
        self.labelList = label

        shuffled_label = label[shuffled_indices]
        
        self.origin_size = [data.shape[1], data.shape[2]]
        
        index_list = np.zeros((0), dtype=np.int64)
        for i in range(9):
            print('load class %d'%i)
            num = (shuffled_label==i).sum()
            num = num // 5
            
            index = np.where(shuffled_label == i)[0]
            test_index = index[num*flag:num*(flag+1)]
            train_index = np.array(list(set(index) - set(test_index)))

            num_val = len(test_index) // 2
            if self.mode == 'train':
                index_list = np.concatenate((index_list, shuffled_indices[train_index]), axis=0)


            elif self.mode == 'val':
                index_list = np.concatenate((index_list, shuffled_indices[test_index[:num_val]]), axis=0)
            else:
                index_list = np.concatenate((index_list, shuffled_indices[test_index[num_val:]]), axis=0)

        self.index_list = index_list
        
        
        self.concept_label_map = [
            [0, 0, 0, 0, 0], # 
            [1, 1, 1, 1, 1], # 
            [2, 2, 2, 2, 2], # BCC
            [3, 3, 3 , 3, 3], # AKIEC
            [4, 4, 4, 4, 4], # BKL
            [5, 5, 5, 5, 5], # DF
            [6, 6, 6, 6, 6], # VASC
            [7, 7, 7, 7, 7],
            [8, 8, 7, 7, 8]
        ]

        print('load done')
    
    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        real_idx = self.index_list[idx]
        img = self.dataList[real_idx]
        label = int(self.labelList[real_idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.return_concept_label:
            return img, label, np.array(self.concept_label_map[label])
        else:
            return img, label
   
    def random_deform(self, img):
        
        img = img.unsqueeze(0)

        dx = (torch.randn_like(img[0, 0, :, :])*2-1) * self.args.deform_scale
        dy = (torch.randn_like(img[0, 0, :, :])*2-1)  * self.args.deform_scale

        dx = self.smooth(dx.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()
        dy = self.smooth(dy.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()


        x_line = torch.linspace(-1, 1, steps=img.shape[2]).unsqueeze(0)
        x_line = x_line.repeat(img.shape[2], 1)
        y_line = torch.linspace(-1, 1, steps=img.shape[3]).unsqueeze(1)
        y_line = y_line.repeat(1, img.shape[3])

        x_line += dx
        y_line += dy

        grid = torch.stack((x_line, y_line), dim=2)
        grid = grid.unsqueeze(0)

        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
    
        img = img.squeeze(0)

        return img


    def random_affine(self, img):
        
        img = img.unsqueeze(0)

        scale_x = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        scale_y = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        
        shear_x = np.random.random() * self.args.scale
        shear_y = np.random.random() * self.args.scale
        #shear_x = 0
        #shear_y = 0


        angle = np.random.randint(-self.args.angle, self.args.angle)
        angle = (angle / 180.) * math.pi

        theta_scale = torch.tensor([[scale_x, shear_x, 0],
                                    [shear_y, scale_y, 0],
                                    [0, 0, 1]]).float()
        theta_rotate = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                    [math.sin(angle), math.cos(angle), 0],
                                    [0, 0, 1]]).float()

        theta = torch.mm(theta_scale, theta_rotate)[0:2, :]
        grid = F.affine_grid(theta.unsqueeze(0), img.size(), align_corners=True)
        
        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        img = img.squeeze(0)

        return img




class siimDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', transforms=None, flag=0, debug=False, config=None, return_concept_label=False):
        self.mode = mode
        if debug:
            dataset_dir = dataset_dir + 'debug'
        print(dataset_dir)
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.flag = flag
        self.args = config
        self.return_concept_label = return_concept_label
        
       
        print('start loading %s dataset'%mode)
        
        data = np.load(dataset_dir+'dataList.npy', mmap_mode='r', allow_pickle=False)
        label = np.load(dataset_dir+'labelList.npy', mmap_mode='r', allow_pickle=False)
        
        rng = np.random.default_rng(29)
        shuffled_indices = rng.permutation(data.shape[0])
        
        self.dataList = data
        self.labelList = label

        shuffled_label = label[shuffled_indices]
        
        self.origin_size = [data.shape[1], data.shape[2]]
        
        index_list = np.zeros((0), dtype=np.int64)
        for i in range(2):
            print('load class %d'%i)
            num = (shuffled_label==i).sum()
            num = num // 5
            
            index = np.where(shuffled_label == i)[0]
            test_index = index[num*flag:num*(flag+1)]
            train_index = np.array(list(set(index) - set(test_index)))

            num_val = len(test_index) // 2
            if self.mode == 'train':
                index_list = np.concatenate((index_list, shuffled_indices[train_index]), axis=0)

            elif self.mode == 'val':
                index_list = np.concatenate((index_list, shuffled_indices[test_index[:num_val]]), axis=0)
            else:
                index_list = np.concatenate((index_list, shuffled_indices[test_index[num_val:]]), axis=0)

        self.index_list = index_list
        
        
        self.concept_label_map = [
            [0, 0, 0, 0, 0, 0, 0], 
            [1, 1, 1, 1, 1, 1, 1], 
        ]

        print('load done')
    
    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        real_idx = self.index_list[idx]
        img = self.dataList[real_idx]
        label = int(self.labelList[real_idx])
        
        # 确保图像数据类型正确
        if isinstance(img, np.ndarray) and img.dtype == np.float32:
            # 如果图像值范围是 [0, 1]
            if img.min() >= 0 and img.max() <= 1:
                img = (img * 255).astype(np.uint8)  # 转换为 [0, 255] 范围内的 uint8 类型
            else:
                # 如果图像值范围不是 [0, 1]，则先进行归一化再转换
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.return_concept_label:
            return img, label, np.array(self.concept_label_map[label])
        else:
            return img, label
   
    def random_deform(self, img):
        
        img = img.unsqueeze(0)

        dx = (torch.randn_like(img[0, 0, :, :])*2-1) * self.args.deform_scale
        dy = (torch.randn_like(img[0, 0, :, :])*2-1)  * self.args.deform_scale

        dx = self.smooth(dx.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()
        dy = self.smooth(dy.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()


        x_line = torch.linspace(-1, 1, steps=img.shape[2]).unsqueeze(0)
        x_line = x_line.repeat(img.shape[2], 1)
        y_line = torch.linspace(-1, 1, steps=img.shape[3]).unsqueeze(1)
        y_line = y_line.repeat(1, img.shape[3])

        x_line += dx
        y_line += dy

        grid = torch.stack((x_line, y_line), dim=2)
        grid = grid.unsqueeze(0)

        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
    
        img = img.squeeze(0)

        return img


    def random_affine(self, img):
        
        img = img.unsqueeze(0)

        scale_x = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        scale_y = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        
        shear_x = np.random.random() * self.args.scale
        shear_y = np.random.random() * self.args.scale
        #shear_x = 0
        #shear_y = 0


        angle = np.random.randint(-self.args.angle, self.args.angle)
        angle = (angle / 180.) * math.pi

        theta_scale = torch.tensor([[scale_x, shear_x, 0],
                                    [shear_y, scale_y, 0],
                                    [0, 0, 1]]).float()
        theta_rotate = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                    [math.sin(angle), math.cos(angle), 0],
                                    [0, 0, 1]]).float()

        theta = torch.mm(theta_scale, theta_rotate)[0:2, :]
        grid = F.affine_grid(theta.unsqueeze(0), img.size(), align_corners=True)
        
        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        img = img.squeeze(0)

        return img
