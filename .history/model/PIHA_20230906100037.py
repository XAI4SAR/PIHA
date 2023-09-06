import torch
import torch.nn as nn
class identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(identity, self).__init__()

    def forward(self, *args):
        return args[0]

class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    def forward_hook(self, module, input, output):
        self.attention = output

class Selective_AvgPool2d(nn.Module):
    def __init__(self, thresh=0.1):
        super(Selective_AvgPool2d, self).__init__()
        self.thresh = thresh
    def forward(self, x):
        x_ = abs(x)>self.thresh
        return ((x*x_).sum(dim=-1).sum(dim=-1)/(x_.sum(dim=-1).sum(dim=-1)+0.000001)).unsqueeze(-1).unsqueeze(-1)
    
class PIHA(nn.Module):
    def __init__(self, part_num, in_channel, down_rate, reduction=2):
        super(PIHA, self).__init__()
        self.part_num = part_num
        self.conv_S1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv_S2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)

        self.phy_group_conv = nn.Conv2d(part_num, in_channel, groups=part_num, kernel_size=down_rate+1, stride=down_rate, padding=1)
        self.se_S1 = SE_Block(in_channel, 2)  
        self.se_S2 = nn.Sequential(
                Selective_AvgPool2d(0.05),
                nn.Conv2d(in_channel//part_num, in_channel// (part_num*reduction),kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel// (part_num*reduction), in_channel//part_num,kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input, ASC_part):
        b, c, h, w = input.size()
        #data-driven stream
        X1 = self.conv_S1(input)
        out1 = self.se_S1(X1)
        #physics-driven steam
        #PAM
        ASC_part_ = self.phy_group_conv(ASC_part)
        X2 = self.conv_S2(input)
        fuse_ = ASC_part_*X2
        fuse = fuse_.view(b, self.part_num, c//self.part_num, h, w) #bs,s,ci,h,w
        #PIA
        se_out=[]
        for idx in range(self.part_num):
            se_out.append(self.se_S2(fuse[:,idx,:,:,:]))
        
        SE_out = torch.stack(se_out,dim=1)

        attention_vectors = self.softmax(SE_out)

        out2 = fuse_*attention_vectors.view(b,-1,1,1)

        return out1 + out2
    def forward_hook(self, module, input, output):
        self.attention = output