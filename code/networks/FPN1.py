import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)





class FPN1(nn.Module):
    def __init__(self):
        super(FPN1, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.upconv = nn.ConvTranspose2d(256,4)
        # Bottom up stages
        self.layer1=nn.Conv2d(64,128,kernel_size=7,stride=2,padding=3,bias=False)
        self.b1=nn.BatchNorm2d(128)

        self.layer2 = nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.b2 = nn.BatchNorm2d(256)

        self.layer3 = nn.Conv2d(256, 512, kernel_size=7, stride=2, padding=3, bias=False)
        self.b3 = nn.BatchNorm2d(512)

        self.layer4 = nn.Conv2d(512,1024, kernel_size=7, stride=2, padding=3, bias=False)
        self.b4 = nn.BatchNorm2d(1024)



        # Top layer
        self.toplayer = conv1x1(1024, 64)

        # Lateral layers
        self.laterallayer1 = conv1x1(512, 64)
        self.laterallayer2 = conv1x1( 256, 64)
        self.laterallayer3 = conv1x1( 128, 64)
        self.laterallayer4 = conv1x1( 64 ,64)

        # Final conv layers
        self.finalconv1 = conv3x3(64, 3)
        self.finalconv2 = conv3x3(64, 3)
        self.finalconv3 = conv3x3(64, 3)
        self.finalconv4 = conv3x3(64, 3)




    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest') + y

    def forward(self, x):
        # Bottom-Up
        c1 = self.relu(self.bn1(self.conv1(x)))

        c2 = self.relu(self.b1(self.layer1(c1)))
        c3 = self.relu(self.b2(self.layer2(c2)))
        c4 = self.relu(self.b3(self.layer3(c3)))
        c5 = self.relu(self.b4(self.layer4(c4)))
        # Top layer && Top-Down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.laterallayer1(c4))
        p3 = self._upsample_add(p4, self.laterallayer2(c3))
        p2 = self._upsample_add(p3, self.laterallayer3(c2))
        p1 = self._upsample_add(p2, self.laterallayer4(c1))
        # Final conv layers

        p4 = self.finalconv4(p4)
        p3 = self.finalconv3(p3)
        p2 = self.finalconv2(p2)
        p1 = self.finalconv1(p1)





        return p4,p3,p2,p1


# def FPN101():
#     return FPN()
#
#
# def test():
#     net = FPN101()
#     feature_maps = net(Variable(torch.randn(4, 1, 224, 224)))
#     for f in feature_maps:
#         print(f.size())
#
# test()