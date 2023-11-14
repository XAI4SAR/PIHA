import torch
import torch.nn as nn
from model.PIHA import *
class Aconvnet_PIHA(nn.Module):
    def __init__(self, num_class, part_num, attention_setting):
        super(Aconvnet_PIHA, self).__init__()
        
        self.cls_conv0 = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.cls_conv1 = nn.Sequential(
            nn.Conv2d(8, 16, 5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        inchannel = 16
        down_rate = 4
        if attention_setting:
            self.phy_attn1 = PIHA(part_num, inchannel, down_rate, reduction=2)
        else:
            self.phy_attn1 = identity()
            
 
        self.cls_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        inchannel = 32
        down_rate = 8
        if attention_setting:
            self.phy_attn2 = PIHA(part_num, inchannel, down_rate, reduction=2)
        else:
            self.phy_attn2 = identity()

        self.cls_conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        inchannel = 64
        down_rate = 16

        if attention_setting:
            self.phy_attn3 = PIHA(part_num, inchannel, down_rate, reduction=2)
        else:
            self.phy_attn3 = identity()
       
        self.cls_conv4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, num_class, 4, stride=1),
            nn.BatchNorm2d(num_class),
        ) 

    def forward(self, x_cls, ASC_part):

        x_cls = self.cls_conv0(x_cls)

        x_cls = self.cls_conv1(x_cls)
        x_cls = self.phy_attn1(x_cls, ASC_part)

        x_cls = self.cls_conv2(x_cls)
        x_cls = self.phy_attn2(x_cls, ASC_part)

        x_cls = self.cls_conv3(x_cls)
        x_cls = self.phy_attn3(x_cls, ASC_part)

        result = self.cls_conv4(x_cls)
        return torch.squeeze(torch.squeeze(result, 2), 2)
    