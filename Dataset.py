from torch.utils.data import Dataset
import numpy as np

class Mstar_Components(Dataset):
    def __init__(self, list_dir, transform=None, part_name='ASC_part'): 
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
    
        data = np.load(npz_path)
        mag_img = abs(data['comp']).squeeze()
        ASC_part = abs(data[self.part_name])

        if self.transform:
            mag_img = self.transform(mag_img)
            ASC_part = self.transform(ASC_part)


        return mag_img, ASC_part, self.data_list['label_list'][idx], npz_path


    def __len__(self):
        return len(self.data_list['npz_list']) #