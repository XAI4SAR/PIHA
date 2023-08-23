import torch
import torch.nn as nn
import sys
from model.PASE import *
sys.path.append('/home/hzl/STAT/code_wu/Experiment') 

class MSNet_basic_block_2(nn.Module):
    def __init__(self, in_channel, out_channel, part_num, down_rate, attention_setting=-1):
        super(MSNet_basic_block_2, self).__init__()
        self.stream_A = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),  
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )



        self.stream_B = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(out_channel),  
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stream_C = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(out_channel),  
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        if attention_setting == -1:
            self.attn_A = identity()
            self.attn_B = identity()
            self.attn_C = identity()
        elif attention_setting == 8:
            self.attn_A = PASE(part_num, out_channel, down_rate, 2)
            self.attn_B = PASE(part_num, out_channel, down_rate, 2)
            self.attn_C = PASE(part_num, out_channel, down_rate, 2)

   
    def forward(self, x, ASC_part): 
        x_A = self.stream_A(x)   
        x_A = self.attn_A(x_A, ASC_part)
        x_B = self.stream_B(x)   
        x_B = self.attn_B(x_B, ASC_part)
        x_C = self.stream_C(x)   
        x_C = self.attn_C(x_C, ASC_part)
        Concat = torch.cat((x_A, x_B, x_C), 1)
        return Concat
#使用第二版的通道注意力，通道上使用卷积分组
class MSNet_last_block_2(nn.Module):
    def __init__(self, in_channel, out_channel, part_num, down_rate, attention_setting=-1):
        super(MSNet_last_block_2, self).__init__()
        self.stream_A = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),     
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.stream_B = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(out_channel),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channel, out_channel, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.stream_C = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(out_channel),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channel, out_channel, kernel_size=11 , stride=1, padding=5),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        if attention_setting == 0:
            self.attn_A = identity()
            self.attn_B = identity()
            self.attn_C = identity()
        elif attention_setting == 1:
            self.attn_A = PASE(part_num, out_channel, down_rate, 2)
            self.attn_B = PASE(part_num, out_channel, down_rate, 2)
            self.attn_C = PASE(part_num, out_channel, down_rate, 2)
        
            
    def forward(self, x, ASC_part): 
        x_A = self.stream_A(x)   
        x_A = self.attn_A(x_A, ASC_part)
        x_B = self.stream_B(x)   
        x_B = self.attn_B(x_B, ASC_part)
        x_C = self.stream_C(x)   
        x_C = self.attn_C(x_C, ASC_part)
        Concat = torch.cat((x_A, x_B, x_C), 1)
        return Concat

#使用第二版的通道注意力，通道上使用卷积分组
class MSNet_phy_attn_2(nn.Module):

    def __init__(self, num_class, part_num, channel, attention_setting):
        super(MSNet_phy_attn_2, self).__init__()

        self.MS1 = MSNet_basic_block_2(1, channel, part_num, 2, attention_setting[0])
        self.MS2 = MSNet_basic_block_2(3*channel, channel, part_num, 4, attention_setting[1])
        self.MS3 = MSNet_last_block_2(3*channel, channel, part_num, 16, attention_setting[2])
        self.Conv_Fu = nn.Conv2d(3*channel, 128, kernel_size=4, stride=1, padding=0)
        self.BN_Fu   = nn.BatchNorm2d(128)
        self.ReLU_Fu = nn.ReLU()
        
        #----------------Full Connection Layers----------------#
        self.FC1     = nn.Linear(128, num_class)
        self.ReLU_FC1 = nn.ReLU()                  

    def forward(self, x, ASC_part):
        x = self.MS1(x, ASC_part)
        x = self.MS2(x, ASC_part)
        x = self.MS3(x, ASC_part)

        x  = self.Conv_Fu(x)                   # 1 X 1
        x  = self.BN_Fu(x)
        x  = self.ReLU_Fu(x)

        #----------------FuLL Connection Layers--------------------------#
        x = x.reshape(x.size(0), -1)
        x = self.FC1(x)                               # 128-10
        x = self.ReLU_FC1(x)
        
        return x