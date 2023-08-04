'''
Author: your name
Date: 2022-03-27 14:52:36
LastEditTime: 2022-11-15 10:02:50
LastEditors: Please set LastEditors
Description: 鎵撳紑koroFileHeader鏌ョ湅閰嶇疆 杩涜璁剧疆: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /Experiment/Experiment1/ASC_BOVW.py
'''

import torch
from torch.utils.data import Dataset
from PIL import Image 
import sys
sys.path.append('/home/hzl/STAT/code_wu/Experiment') 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
import cv2

import os
from torchvision import transforms

class Mstar_ASC_part(Dataset):
    def __init__(self, list_dir, transform=None): #
        self.data_list = {'npz_list':[], 'label_list':[]}  
        self.transform = transform
        f = open(list_dir, 'r')
        for i in f.readlines():
            self.data_list['npz_list'].append(i.strip().split()[0])
            self.data_list['label_list'].append(int(i.strip().split()[1]))

    def __getitem__(self, idx):
        
        npz_path = self.data_list['npz_list'][idx]
    
        # print(npz_path)
        data = np.load(npz_path)
        mag_img = abs(data['comp']).squeeze()
        ASC_part_x_y_A = abs(data['ASC_part_x_y_A'])
        ASC_part_L_phi_a_A = abs(data['ASC_part_L_phi_a_A']) 
        # shadow = abs(data['shadow']) 
        if self.transform:
            mag_img = self.transform(mag_img)
        ASC_part_x_y_A = transforms.ToTensor()(ASC_part_x_y_A)
        ASC_part_L_phi_a_A = transforms.ToTensor()(ASC_part_L_phi_a_A)
        return mag_img, ASC_part_x_y_A, ASC_part_L_phi_a_A, self.data_list['label_list'][idx], npz_path


    def __len__(self):
        return len(self.data_list['npz_list']) #

class Mstar_multi_ASC_part(Dataset):
    def __init__(self, list_dir, dataset_name, transform=None): #
        self.data_list = {'npz_list':[], 'label_list':[]}  
        self.dataset_name = dataset_name
        self.transform = transform
        f = open(list_dir, 'r')
        for i in f.readlines():
            self.data_list['npz_list'].append(i.strip().split()[0])
            self.data_list['label_list'].append(int(i.strip().split()[1]))

    def __getitem__(self, idx):
        
        npz_path_1 = self.data_list['npz_list'][idx]
        npz_path_2 = npz_path_1.replace(self.dataset_name[0], self.dataset_name[1])
        npz_path_3 = npz_path_1.replace(self.dataset_name[0], self.dataset_name[2])
        # print(npz_path)
        data_1 = np.load(npz_path_1)
        data_2 = np.load(npz_path_2)
        data_3 = np.load(npz_path_3)
        mag_img = abs(data_1['comp']+data_1['ISAR']).squeeze()
        ASC_part_x_y_A_1 = abs(data_1['ASC_part_x_y_A'])
        ASC_part_L_phi_a_A_1 = abs(data_1['ASC_part_L_phi_a_A']) 

        ASC_part_x_y_A_2 = abs(data_2['ASC_part_x_y_A'])
        ASC_part_L_phi_a_A_2 = abs(data_2['ASC_part_L_phi_a_A']) 

        ASC_part_x_y_A_3 = abs(data_3['ASC_part_x_y_A'])
        ASC_part_L_phi_a_A_3 = abs(data_3['ASC_part_L_phi_a_A']) 
        # shadow = abs(data['shadow']) 
        if self.transform:
            mag_img = self.transform(mag_img)
        ASC_part_x_y_A_1 = transforms.ToTensor()(ASC_part_x_y_A_1)
        ASC_part_L_phi_a_A_1 = transforms.ToTensor()(ASC_part_L_phi_a_A_1)
        ASC_part_x_y_A_2 = transforms.ToTensor()(ASC_part_x_y_A_2)
        ASC_part_L_phi_a_A_2 = transforms.ToTensor()(ASC_part_L_phi_a_A_2)
        ASC_part_x_y_A_3 = transforms.ToTensor()(ASC_part_x_y_A_3)
        ASC_part_L_phi_a_A_3 = transforms.ToTensor()(ASC_part_L_phi_a_A_3)
        return mag_img, ASC_part_x_y_A_1, ASC_part_L_phi_a_A_1, ASC_part_x_y_A_2, ASC_part_L_phi_a_A_2, ASC_part_x_y_A_3, ASC_part_L_phi_a_A_3, self.data_list['label_list'][idx]


    def __len__(self):
        return len(self.data_list['npz_list']) #
    

class Mstar(Dataset):
    def __init__(self, list_dir, transform=None): #
        self.data_list = {'npz_list':[], 'label_list':[]}  
        self.transform = transform
        f = open(list_dir, 'r')
        for i in f.readlines():
            self.data_list['npz_list'].append(i.strip().split()[0])
            self.data_list['label_list'].append(int(i.strip().split()[1]))

    def __getitem__(self, idx):
        npz_path = self.data_list['npz_list'][idx]

        data = np.load(npz_path)
        mag_img = abs(data['comp']).squeeze()

        # shadow = abs(data['shadow']) 
        if self.transform:
            mag_img = self.transform(mag_img)
        return mag_img, self.data_list['label_list'][idx], npz_path


    def __len__(self):
        return len(self.data_list['npz_list']) 

class Mstar_ASC_part_shadow(Dataset):
    def __init__(self, list_dir, transform=None): #
        self.data_list = {'npz_list':[], 'label_list':[]}  
        self.transform = transform
        f = open(list_dir, 'r')
        for i in f.readlines():
            self.data_list['npz_list'].append(i.strip().split()[0])
            self.data_list['label_list'].append(int(i.strip().split()[1]))

    def __getitem__(self, idx):
        npz_path = self.data_list['npz_list'][idx]

        data = torch.load(npz_path)
        mag_img = abs(data['comp'])
        ASC_part_x_y_A = abs(data['ASC_part_x_y_A'])
        ASC_part_L_phi_a_A = abs(data['ASC_part_L_phi_a_A']) 
        shadow = abs(data['shadow']) 
        # cv2.imwrite('2.jpg', shadow.detach().cpu().numpy().squeeze()*255)
        blank = torch.zeros((1, 128, 128))
        if shadow.sum().item()==0:
            print(npz_path)
        min_vals = torch.min(mag_img[shadow.bool()])
        max_vals = torch.max(mag_img[shadow.bool()])
 
        # 最小-最大缩放，将x的范围缩放到[0, 1]
        # plt.subplot(1,3,1)
        # plt.imshow((mag_img).detach().cpu().numpy().squeeze())
        # plt.subplot(1,3,2)
        # plt.imshow((shadow).detach().cpu().numpy().squeeze())
        blank[shadow.bool()] = 10*mag_img.mean()*(mag_img[shadow.bool()] - min_vals) / (max_vals - min_vals)
        out_img = torch.cat((mag_img, blank), dim=0)
        # plt.subplot(1,3,3)
        # plt.imshow((blank).detach().cpu().numpy().squeeze())
        # plt.savefig('1.jpg')
        if self.transform: 
            out_img = self.transform(out_img)
            ASC_part_x_y_A = self.transform(ASC_part_x_y_A)
            ASC_part_L_phi_a_A = self.transform(ASC_part_L_phi_a_A)
        return out_img, ASC_part_x_y_A, ASC_part_L_phi_a_A, shadow, self.data_list['label_list'][idx], npz_path


    def __len__(self):
        return len(self.data_list['npz_list']) #

class Mstar_ASC_part_2(Dataset):
    def __init__(self, list_dir, transform=None, part_name='ASC_part'): #
        self.data_list = {'npz_list':[], 'label_list':[]}  
        self.transform = transform
        if part_name:

            self.part_name = part_name
        else:
            self.part_name = 'ASC_part'
        f = open(list_dir, 'r')
        for i in f.readlines():
            self.data_list['npz_list'].append(i.strip().split()[0])
            self.data_list['label_list'].append(int(i.strip().split()[1]))

    def __getitem__(self, idx):
        
        npz_path = self.data_list['npz_list'][idx]
    
        # print(npz_path)
        data = np.load(npz_path)
        mag_img = abs(data['comp']).squeeze()
        ASC_part = abs(data[self.part_name])

        # shadow = abs(data['shadow']) 
        if self.transform:
            mag_img = self.transform(mag_img)
            ASC_part = self.transform(ASC_part)


        return mag_img, ASC_part, self.data_list['label_list'][idx], npz_path


    def __len__(self):
        return len(self.data_list['npz_list']) #
    
class Mstar_ASC_part_analyze(Dataset):
    def __init__(self, list_dir, transform=None, part_name='ASC_part'): #
        self.data_list = {'npz_list':[], 'label_list':[]}  
        if part_name:

            self.part_name = part_name
        else:
            self.part_name = 'ASC_part'
        self.transform = transform
        f = open(list_dir, 'r')
        for i in f.readlines():
            self.data_list['npz_list'].append(i.strip().split()[0])
            self.data_list['label_list'].append(int(i.strip().split()[1]))

    def __getitem__(self, idx):
        
        npz_path = self.data_list['npz_list'][idx]
    
        # print(npz_path)
        data = np.load(npz_path)
        mag_img = abs(data['comp']).squeeze()
        ASC_part = abs(data[self.part_name])
        Target_Az = data['TargetAz']
        # shadow = abs(data['shadow']) 
        if self.transform:
            mag_img = self.transform(mag_img)
            ASC_part = self.transform(ASC_part)

        return mag_img, ASC_part, self.data_list['label_list'][idx], Target_Az, npz_path


    def __len__(self):
        return len(self.data_list['npz_list']) #
if __name__ == '__main__':
    list_path = '/home/hzl/STAT/code_wu/Experiment/Experiment1/MSTAR_EXP/10CLASS_2/LIST/test.txt'
    dataset = Mstar_ASC_part(list_path, None, [300, 360])
    print(len(dataset))