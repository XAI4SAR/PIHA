import torch
import torch.nn as nn
import sys
from model.PASE import *
sys.path.append('/home/hzl/STAT/code_wu/Experiment') 
class MSNet_basic_block(nn.Module):
    def __init__(self, in_channel, out_channel, input_size, part_num, attn=-1):
        super(MSNet_basic_block, self).__init__()
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

        if attn == -1:
            self.attn_A = identity()
            self.attn_B = identity()
            self.attn_C = identity()
        elif attn == 0:
            self.attn_A = Phy_Attention_channel(input_size, part_num, out_channel)
            self.attn_B = Phy_Attention_channel(input_size, part_num, out_channel)
            self.attn_C = Phy_Attention_channel(input_size, part_num, out_channel)
        elif attn == 1:
            self.attn_A = Phy_Attention_1_2(input_size, part_num)
            self.attn_B = Phy_Attention_1_2(input_size, part_num)
            self.attn_C = Phy_Attention_1_2(input_size, part_num)
       
    def forward(self, x, ASC_part_x_y_A, ASC_part_L_phi_a_A): 
        x_A = self.stream_A(x)   
        x_A = self.attn_A(x_A, ASC_part_x_y_A, ASC_part_L_phi_a_A)
        x_B = self.stream_B(x)   
        x_B = self.attn_B(x_B, ASC_part_x_y_A, ASC_part_L_phi_a_A)
        x_C = self.stream_C(x)   
        x_C = self.attn_C(x_C, ASC_part_x_y_A, ASC_part_L_phi_a_A)
        Concat = torch.cat((x_A, x_B, x_C), 1)
        return Concat

class MSNet_last_block(nn.Module):
    def __init__(self, in_channel, out_channel, input_size, part_num, attn=-1 ):
        super(MSNet_last_block, self).__init__()
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

        if attn == -1:
            self.attn_A = identity()
            self.attn_B = identity()
            self.attn_C = identity()
        elif attn == 0:
            self.attn_A = Phy_Attention_channel(input_size, part_num, out_channel)
            self.attn_B = Phy_Attention_channel(input_size, part_num, out_channel)
            self.attn_C = Phy_Attention_channel(input_size, part_num, out_channel)
        elif attn == 1:
            self.attn_A = Phy_Attention_1_2(input_size, part_num)
            self.attn_B = Phy_Attention_1_2(input_size, part_num)
            self.attn_C = Phy_Attention_1_2(input_size, part_num)

    def forward(self, x, ASC_part_x_y_A, ASC_part_L_phi_a_A): 
        x_A = self.stream_A(x)   
        x_A = self.attn_A(x_A, ASC_part_x_y_A, ASC_part_L_phi_a_A)
        x_B = self.stream_B(x)   
        x_B = self.attn_B(x_B, ASC_part_x_y_A, ASC_part_L_phi_a_A)
        x_C = self.stream_C(x)   
        x_C = self.attn_C(x_C, ASC_part_x_y_A, ASC_part_L_phi_a_A)
        Concat = torch.cat((x_A, x_B, x_C), 1)
        return Concat
    
class MSNet_phy_attn(nn.Module):

    def __init__(self, num_class, part_num, attention_setting, channel):
        super(MSNet_phy_attn, self).__init__()

        self.MS1 = MSNet_basic_block(1, channel, 32, part_num, attention_setting[0])
        self.MS2 = MSNet_basic_block(3*channel, channel, 16, part_num, attention_setting[1])
        self.MS3 = MSNet_last_block(3*channel, channel, 4, part_num, attention_setting[2])
        self.Conv_Fu = nn.Conv2d(3*channel, 128, kernel_size=4, stride=1, padding=0)
        self.BN_Fu   = nn.BatchNorm2d(128)
        self.ReLU_Fu = nn.ReLU()
        
        #----------------Full Connection Layers----------------#
        self.FC1     = nn.Linear(128, num_class)
        self.ReLU_FC1 = nn.ReLU()                  

    def forward(self, x, ASC_part_x_y_A, ASC_part_L_phi_a_A):
        x = self.MS1(x, ASC_part_x_y_A, ASC_part_L_phi_a_A)
        x = self.MS2(x, ASC_part_x_y_A, ASC_part_L_phi_a_A)
        x = self.MS3(x, ASC_part_x_y_A, ASC_part_L_phi_a_A)

        x  = self.Conv_Fu(x)                   # 1 X 1
        x  = self.BN_Fu(x)
        x  = self.ReLU_Fu(x)

        #----------------FuLL Connection Layers--------------------------#
        x = x.reshape(x.size(0), -1)
        x = self.FC1(x)                               # 128-10
        x = self.ReLU_FC1(x)
        
        return x

class MSNet_basic_SE_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSNet_basic_SE_block, self).__init__()
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

        self.attn_A = SE_Block(out_channel)
        self.attn_B = SE_Block(out_channel)
        self.attn_C = SE_Block(out_channel)

        
    def forward(self, x): 
        x_A = self.stream_A(x)   
        x_A = self.attn_A(x_A)
        x_B = self.stream_B(x)   
        x_B = self.attn_B(x_B)
        x_C = self.stream_C(x)   
        x_C = self.attn_C(x_C)
        Concat = torch.cat((x_A, x_B, x_C), 1)
        return Concat

class MSNet_last_SE_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSNet_last_SE_block, self).__init__()
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


        self.attn_A = SE_Block(out_channel)
        self.attn_B = SE_Block(out_channel)
        self.attn_C = SE_Block(out_channel)

        
    def forward(self, x): 
        x_A = self.stream_A(x)   
        x_A = self.attn_A(x_A)
        x_B = self.stream_B(x)   
        x_B = self.attn_B(x_B)
        x_C = self.stream_C(x)   
        x_C = self.attn_C(x_C)
        Concat = torch.cat((x_A, x_B, x_C), 1)
        return Concat
    
class MSNet_SE(nn.Module):

    def __init__(self, num_class, channel):
        super(MSNet_SE, self).__init__()

        self.MS1 = MSNet_basic_SE_block(1, channel)
        self.MS2 = MSNet_basic_SE_block(3*channel, channel)
        self.MS3 = MSNet_last_SE_block(3*channel, channel)
        self.Conv_Fu = nn.Conv2d(3*channel, 3*channel, kernel_size=4, stride=1, padding=0)
        self.BN_Fu   = nn.BatchNorm2d(3*channel)
        self.ReLU_Fu = nn.ReLU()
        
        #----------------Full Connection Layers----------------#
        self.FC1     = nn.Linear(3*channel, num_class)
        self.ReLU_FC1 = nn.ReLU()                  

    def forward(self, x):
        x = self.MS1(x)
        x = self.MS2(x)
        x = self.MS3(x)

        x  = self.Conv_Fu(x)                   # 1 X 1
        x  = self.BN_Fu(x)
        x  = self.ReLU_Fu(x)

        #----------------FuLL Connection Layers--------------------------#
        x = x.reshape(x.size(0), -1)
        x = self.FC1(x)                               # 128-10
        x = self.ReLU_FC1(x)
        
        return x
    
class MSNet_basic_multi_phy_block(nn.Module):
    def __init__(self, in_channel, out_channel, input_size, part_num, attn=-1):
        super(MSNet_basic_multi_phy_block, self).__init__()
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

        if attn == -1:
            self.attn_A = identity()
            self.attn_B = identity()
            self.attn_C = identity()
        elif attn == 0:
            self.attn_A = Phy_Attention_channel(input_size, part_num[0], out_channel)
            self.attn_B = Phy_Attention_channel(input_size, part_num[1], out_channel)
            self.attn_C = Phy_Attention_channel(input_size, part_num[2], out_channel)
        elif attn == 1:
            self.attn_A = Phy_Attention_1_2(input_size, part_num[0])
            self.attn_B = Phy_Attention_1_2(input_size, part_num[1])
            self.attn_C = Phy_Attention_1_2(input_size, part_num[2])
     
    def forward(self, x, ASC_part_x_y_A_1, ASC_part_L_phi_a_A_1, ASC_part_x_y_A_2, ASC_part_L_phi_a_A_2, ASC_part_x_y_A_3, ASC_part_L_phi_a_A_3): 
        x_A = self.stream_A(x)   
        x_A = self.attn_A(x_A, ASC_part_x_y_A_1, ASC_part_L_phi_a_A_1)
        x_B = self.stream_B(x)   
        x_B = self.attn_B(x_B, ASC_part_x_y_A_2, ASC_part_L_phi_a_A_2)
        x_C = self.stream_C(x)   
        x_C = self.attn_C(x_C, ASC_part_x_y_A_3, ASC_part_L_phi_a_A_3)
        Concat = torch.cat((x_A, x_B, x_C), 1)
        return Concat

class MSNet_last_multi_phy_block(nn.Module):
    def __init__(self, in_channel, out_channel, input_size, part_num, attn=-1 ):
        super(MSNet_last_multi_phy_block, self).__init__()
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

        if attn == -1:
            self.attn_A = identity()
            self.attn_B = identity()
            self.attn_C = identity()
        elif attn == 0:
            self.attn_A = Phy_Attention_channel(input_size, part_num[0], out_channel)
            self.attn_B = Phy_Attention_channel(input_size, part_num[1], out_channel)
            self.attn_C = Phy_Attention_channel(input_size, part_num[2], out_channel)
        elif attn == 1:
            self.attn_A = Phy_Attention_1_2(input_size, part_num[0])
            self.attn_B = Phy_Attention_1_2(input_size, part_num[1])
            self.attn_C = Phy_Attention_1_2(input_size, part_num[2])

    def forward(self, x, ASC_part_x_y_A_1, ASC_part_L_phi_a_A_1, ASC_part_x_y_A_2, ASC_part_L_phi_a_A_2, ASC_part_x_y_A_3, ASC_part_L_phi_a_A_3): 
        x_A = self.stream_A(x)   
        x_A = self.attn_A(x_A, ASC_part_x_y_A_1, ASC_part_L_phi_a_A_1)
        x_B = self.stream_B(x)   
        x_B = self.attn_B(x_B, ASC_part_x_y_A_2, ASC_part_L_phi_a_A_2)
        x_C = self.stream_C(x)   
        x_C = self.attn_C(x_C, ASC_part_x_y_A_3, ASC_part_L_phi_a_A_3)
        Concat = torch.cat((x_A, x_B, x_C), 1)
        return Concat
    
class MSNet_multi_phy(nn.Module):

    def __init__(self, num_class, part_num, attention_setting):
        super(MSNet_multi_phy, self).__init__()

        self.MS1 = MSNet_basic_multi_phy_block(1, 40, 32, part_num, attention_setting[0])
        self.MS2 = MSNet_basic_multi_phy_block(120, 40, 16, part_num, attention_setting[1])
        self.MS3 = MSNet_last_multi_phy_block(120, 40, 4, part_num, attention_setting[2])
        self.Conv_Fu = nn.Conv2d(120, 128, kernel_size=4, stride=1, padding=0)
        self.BN_Fu   = nn.BatchNorm2d(128)
        self.ReLU_Fu = nn.ReLU()
        
        #----------------Full Connection Layers----------------#
        self.FC1     = nn.Linear(128, num_class)
        self.ReLU_FC1 = nn.ReLU()                  

    def forward(self, x, ASC_part_x_y_A_1, ASC_part_L_phi_a_A_1, ASC_part_x_y_A_2, ASC_part_L_phi_a_A_2, ASC_part_x_y_A_3, ASC_part_L_phi_a_A_3):
        x = self.MS1(x, ASC_part_x_y_A_1, ASC_part_L_phi_a_A_1, ASC_part_x_y_A_2, ASC_part_L_phi_a_A_2, ASC_part_x_y_A_3, ASC_part_L_phi_a_A_3)
        x = self.MS2(x, ASC_part_x_y_A_1, ASC_part_L_phi_a_A_1, ASC_part_x_y_A_2, ASC_part_L_phi_a_A_2, ASC_part_x_y_A_3, ASC_part_L_phi_a_A_3)
        x = self.MS3(x, ASC_part_x_y_A_1, ASC_part_L_phi_a_A_1, ASC_part_x_y_A_2, ASC_part_L_phi_a_A_2, ASC_part_x_y_A_3, ASC_part_L_phi_a_A_3)

        x  = self.Conv_Fu(x)                   # 1 X 1
        x  = self.BN_Fu(x)
        x  = self.ReLU_Fu(x)

        #----------------FuLL Connection Layers--------------------------#
        x = x.reshape(x.size(0), -1)
        x = self.FC1(x)                               # 128-10
        x = self.ReLU_FC1(x)
        
        return x
    
# class MSNet_phy_attn_2(nn.Module):

#     def __init__(self, num_class, part_num, attention_setting, channel):
#         super(MSNet_phy_attn_2, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),  
#             nn.ReLU(),
#         )
#         self.MS1 = MSNet_basic_block(32, 64, 32, part_num, attention_setting[0])
#         self.MS2 = MSNet_basic_block(192, 128, 16, part_num, attention_setting[1])
#         self.MS3 = MSNet_last_block(384, 256, 4, part_num, attention_setting[2])
#         self.Conv_Fu = nn.Conv2d(768, 128, kernel_size=4, stride=1, padding=0)
#         self.BN_Fu   = nn.BatchNorm2d(128)
#         self.ReLU_Fu = nn.ReLU()
        
#         #----------------Full Connection Layers----------------#
#         self.FC1     = nn.Linear(128, num_class)
#         self.ReLU_FC1 = nn.ReLU()                  

#     def forward(self, x, ASC_part_x_y_A, ASC_part_L_phi_a_A):
#         x = self.conv1(x)
#         x = self.MS1(x, ASC_part_x_y_A, ASC_part_L_phi_a_A)
#         x = self.MS2(x, ASC_part_x_y_A, ASC_part_L_phi_a_A)
#         x = self.MS3(x, ASC_part_x_y_A, ASC_part_L_phi_a_A)

#         x  = self.Conv_Fu(x)                   # 1 X 1
#         x  = self.BN_Fu(x)
#         x  = self.ReLU_Fu(x)

#         #----------------FuLL Connection Layers--------------------------#
#         x = x.reshape(x.size(0), -1)
#         x = self.FC1(x)                               # 128-10
#         x = self.ReLU_FC1(x)
        
#         return x
#使用第二版的通道注意力，通道上使用卷积分组
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
        elif attention_setting == 0:
            self.attn_A = Phy_Attention_2(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2(part_num, out_channel, down_rate, 2)
        elif attention_setting == 1:
            self.attn_A = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)
        elif attention_setting == 2:
            self.attn_A = Phy_Attention_2_3(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_3(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_3(part_num, out_channel, down_rate, 2)
        elif attention_setting == 3:
            self.attn_A = Phy_Attention_2_4(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_4(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_4(part_num, out_channel, down_rate, 2)
        elif attention_setting == 8:
            self.attn_A = Phy_Attention_2_9(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_9(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_9(part_num, out_channel, down_rate, 2)
        elif attention_setting == 10:
            self.attn_A = Phy_Attention_2_11(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_11(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_11(part_num, out_channel, down_rate, 2)
        elif attention_setting == 11:
            self.attn_A = Phy_Attention_2_12(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_12(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_12(part_num, out_channel, down_rate, 2)
   
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

        if attention_setting == -1:
            self.attn_A = identity()
            self.attn_B = identity()
            self.attn_C = identity()
        elif attention_setting == 0:
            self.attn_A = Phy_Attention_2(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2(part_num, out_channel, down_rate, 2)
        elif attention_setting == 1:
            self.attn_A = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)
        elif attention_setting == 2:
            self.attn_A = Phy_Attention_2_3(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_3(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_3(part_num, out_channel, down_rate, 2)
        elif attention_setting == 3:
            self.attn_A = Phy_Attention_2_4(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_4(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_4(part_num, out_channel, down_rate, 2)
        elif attention_setting == 8:
            self.attn_A = Phy_Attention_2_9(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_9(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_9(part_num, out_channel, down_rate, 2)
        elif attention_setting == 10:
            self.attn_A = Phy_Attention_2_11(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_11(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_11(part_num, out_channel, down_rate, 2)
        elif attention_setting == 11:
            self.attn_A = Phy_Attention_2_12(part_num, out_channel, down_rate, 2)
            self.attn_B = Phy_Attention_2_12(part_num, out_channel, down_rate, 2)
            self.attn_C = Phy_Attention_2_12(part_num, out_channel, down_rate, 2)
            
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
    
#使用第三版的通道注意力，通道上使用拆分通道分组
class MSNet_basic_block_3(nn.Module):
    def __init__(self, in_channel, out_channel, part_num, down_rate):
        super(MSNet_basic_block_3, self).__init__()
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


        self.attn_A = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)
        self.attn_B = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)
        self.attn_C = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)
       
    def forward(self, x, ASC_part): 
        x_A = self.stream_A(x)   
        x_A = self.attn_A(x_A, ASC_part)
        x_B = self.stream_B(x)   
        x_B = self.attn_B(x_B, ASC_part)
        x_C = self.stream_C(x)   
        x_C = self.attn_C(x_C, ASC_part)
        Concat = torch.cat((x_A, x_B, x_C), 1)
        return Concat
#使用第三版的通道注意力，通道上使用拆分通道分组
class MSNet_last_block_3(nn.Module):
    def __init__(self, in_channel, out_channel, part_num, down_rate):
        super(MSNet_last_block_3, self).__init__()
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

        self.attn_A = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)
        self.attn_B = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)
        self.attn_C = Phy_Attention_2_2(part_num, out_channel, down_rate, 2)

    def forward(self, x, ASC_part): 
        x_A = self.stream_A(x)   
        x_A = self.attn_A(x_A, ASC_part)
        x_B = self.stream_B(x)   
        x_B = self.attn_B(x_B, ASC_part)
        x_C = self.stream_C(x)   
        x_C = self.attn_C(x_C, ASC_part)
        Concat = torch.cat((x_A, x_B, x_C), 1)
        return Concat

#使用第三版的通道注意力，通道上使用拆分通道分组
class MSNet_phy_attn_3(nn.Module):

    def __init__(self, num_class, part_num, channel):
        super(MSNet_phy_attn_3, self).__init__()

        self.MS1 = MSNet_basic_block_3(1, channel, part_num, down_rate=2)
        self.MS2 = MSNet_basic_block_3(3*channel, channel, part_num, down_rate=4)
        self.MS3 = MSNet_last_block_3(3*channel, channel, part_num, down_rate=16)
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