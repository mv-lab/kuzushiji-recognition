import torch
from torch import nn, optim
import torch.nn.functional as F
import pretrainedmodels


def conv3x3(in_channel, out_channel): #not change resolusion
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=3,stride=1,padding=1,dilation=1,bias=False)

def conv1x1(in_channel, out_channel): #not change resolution
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=1,stride=1,padding=0,dilation=1,bias=False)

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
            
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        

    
#U-Net ResNet34
class UNET_RESNET34(nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()
        
        class CenterBlock(nn.Module):
            def __init__(self, in_channel, out_channel):
                super().__init__()
                self.conv = conv3x3(in_channel, out_channel).apply(init_weight)

            def forward(self, inputs):
                x = self.conv(inputs)
                return x


        class DecodeBlock(nn.Module):
            def __init__(self, in_channel, out_channel, upsample):
                super().__init__()
                self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)
                self.upsample = nn.Sequential()
                if upsample:
                    self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='nearest'))
                self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
                self.bn2 = nn.BatchNorm2d(in_channel).apply(init_weight)
                self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
                self.conv1x1   = conv1x1(in_channel, out_channel).apply(init_weight)

            def forward(self, inputs):
                x  = F.relu(self.bn1(inputs))
                x  = self.upsample(x)
                x  = self.conv3x3_1(x)
                x  = self.conv3x3_2(F.relu(self.bn2(x)))
                x += self.conv1x1(self.upsample(inputs)) #shortcut
                return x
    
    
        #encoder
        if load_weights:
            resnet34 = pretrainedmodels.__dict__['resnet34'](num_classes=1000,pretrained='imagenet')
        else:
            resnet34 = pretrainedmodels.__dict__['resnet34'](num_classes=1000,pretrained=None)
        self.conv1   = resnet34.conv1 #(*,3,h,w)->(*,64,h/2,w/2)
        self.bn1     = resnet34.bn1
        self.maxpool = resnet34.maxpool #->(*,64,h/4,w/4)
        self.layer1  = resnet34.layer1 #->(*,64,h/4,w/4)
        self.layer2  = resnet34.layer2 #->(*,128,h/8,w/8)
        self.layer3  = resnet34.layer3 #->(*,256,h/16,w/16)
        self.layer4  = resnet34.layer4 #->(*,512,h/32,w/32)
        
        #center
        self.center  = CenterBlock(512,512) #->(*,512,h/32,w/32)
        
        #decoder
        self.decoder4 = DecodeBlock(512+512,64, upsample=True) #->(*,64,h/16,w/16)
        self.decoder3 = DecodeBlock(64+256,64, upsample=True) #->(*,64,h/8,w/8)
        self.decoder2 = DecodeBlock(64+128,64,  upsample=True) #->(*,64,h/4,w/4)
        self.decoder1 = DecodeBlock(64+64,64,   upsample=True) #->(*,64,h/2,w/2)
        self.decoder0 = DecodeBlock(64,64, upsample=True) #->(*,64,h,w)
        
        #final conv
        self.final_conv = nn.Sequential(
            conv3x3(64,64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64,3).apply(init_weight) #heatmap, heightmap, widthmap
        )
        
    def forward(self, inputs):
        #encoder
        x0 = F.relu(self.bn1(self.conv1(inputs))) #->(*,64,h/2,w/2)
        x0 = self.maxpool(x0) #->(*,64,h/4,w/4)
        x1 = self.layer1(x0) #->(*,64,h/4,w/4)
        x2 = self.layer2(x1) #->(*,128,h/8,w/8)
        x3 = self.layer3(x2) #->(*,256,h/16,w/16)
        x4 = self.layer4(x3) #->(*,512,h/32,w/32)
        
        #center
        y5 = self.center(x4) #->(*,512,h/32,w/32)
        
        #decoder
        y4 = self.decoder4(torch.cat([x4,y5], dim=1)) #->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3,y4], dim=1)) #->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2,y3], dim=1)) #->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1,y2], dim=1)) #->(*,64,h/2,w/2)
        y0 = self.decoder0(y1) #->(*,64,h,w)
        
        #final conv
        logits = self.final_conv(y0) #->(*,3,h,w)
        return logits
    
    
    
        
class HGNET_RESNET34_Module(nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()
        
        class CenterBlock(nn.Module):
            def __init__(self, in_channel, out_channel):
                super().__init__()
                self.conv = conv3x3(in_channel, out_channel).apply(init_weight)

            def forward(self, inputs):
                x = self.conv(inputs)
                return x


        class DecodeBlock(nn.Module):
            def __init__(self, in_channel, out_channel, upsample):
                super().__init__()
                self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)
                self.upsample = nn.Sequential()
                if upsample:
                    self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='nearest'))
                self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
                self.bn2 = nn.BatchNorm2d(in_channel).apply(init_weight)
                self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
                self.conv1x1   = conv1x1(in_channel, out_channel).apply(init_weight)

            def forward(self, inputs):
                x  = F.relu(self.bn1(inputs))
                x  = self.upsample(x)
                x  = self.conv3x3_1(x)
                x  = self.conv3x3_2(F.relu(self.bn2(x)))
                x += self.conv1x1(self.upsample(inputs)) #shortcut
                return x

        
        #encoder
        if load_weights:
            resnet34 = pretrainedmodels.__dict__['resnet34'](num_classes=1000,pretrained='imagenet')
        else:
            resnet34 = pretrainedmodels.__dict__['resnet34'](num_classes=1000,pretrained=None)
        self.conv1   = resnet34.conv1 #(*,3,h,w)->(*,64,h/2,w/2)
        self.bn1     = resnet34.bn1
        self.maxpool = resnet34.maxpool #->(*,64,h/4,w/4)
        self.layer1  = resnet34.layer1 #->(*,64,h/4,w/4)
        self.layer2  = resnet34.layer2 #->(*,128,h/8,w/8)
        self.layer3  = resnet34.layer3 #->(*,256,h/16,w/16)
        self.layer4  = resnet34.layer4 #->(*,512,h/32,w/32)
        
        #center
        self.center  = CenterBlock(512,512) #->(*,512,h/32,w/32)
        
        #decoder
        self.decoder4 = DecodeBlock(512+512,64, upsample=True) #->(*,64,h/16,w/16)
        self.decoder3 = DecodeBlock(64+256,64, upsample=True) #->(*,64,h/8,w/8)
        self.decoder2 = DecodeBlock(64+128,64,  upsample=True) #->(*,64,h/4,w/4)
        self.decoder1 = DecodeBlock(64+64,64,   upsample=True) #->(*,64,h/2,w/2)
        self.decoder0 = DecodeBlock(64,64, upsample=True) #->(*,64,h,w)
        
        
    def forward(self, inputs):
        #encoder1
        x0 = F.relu(self.bn1(self.conv1(inputs))) #->(*,64,h/2,w/2)
        x0 = self.maxpool(x0) #->(*,64,h/4,w/4)
        x1 = self.layer1(x0) #->(*,64,h/4,w/4)
        x2 = self.layer2(x1) #->(*,128,h/8,w/8)
        x3 = self.layer3(x2) #->(*,256,h/16,w/16)
        x4 = self.layer4(x3) #->(*,512,h/32,w/32)
        
        #center1
        y5 = self.center(x4) #->(*,512,h/32,w/32)
        
        #decoder1
        y4 = self.decoder4(torch.cat([x4,y5], dim=1)) #->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3,y4], dim=1)) #->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2,y3], dim=1)) #->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1,y2], dim=1)) #->(*,64,h/2,w/2)
        y0 = self.decoder0(y1) #->(*,64,h,w)
        
        return y0
    
    
        
#HourglassNet ResNet34
class HGNET_RESNET34(nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()
        self.module1 = HGNET_RESNET34_Module(load_weights)
        self.module2 = HGNET_RESNET34_Module(load_weights)
        
        self.conv1x1 = conv1x1(64,3).apply(init_weight)
        
        #final conv1
        self.final_conv = nn.Sequential(
            conv3x3(64,64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64,3).apply(init_weight)
        )
        
        #final conv2
        self.final_conv = nn.Sequential(
            conv3x3(64,64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64,3).apply(init_weight)
        )
        
    def forward(self, inputs):
        y0_1 = self.module1(inputs)
        #final conv1
        logits1 = self.final_conv(y0_1) #->(*,3,h,w)
        
        y0_2 = self.module2(self.conv1x1(y0_1))
        #final conv2
        logits2 = self.final_conv(y0_2) #->(*,3,h,w)
        return logits1, logits2