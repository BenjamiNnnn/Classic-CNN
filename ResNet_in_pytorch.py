# 导入包
import torch
from torch import nn 
from torch.nn import functional as F
class ResBlk(nn.Module):
    """
    ResNet Block
    """
    expansion = 1
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk, self).__init__()
        """
             两个网络层
             每个网络层 包含 3x3 的卷积
        """
        self.blockfunc = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch_out, ch_out*ResBlk.expansion, kernel_size=3,padding=1),
            nn.BatchNorm2d(ch_out*ResBlk.expansion),
        )

        self.extra = nn.Sequential()
        if stride!=1 or ch_out*ResBlk.expansion != ch_in:
            # [b,ch_in,h,w] => [b,ch_out,h,w]
            self.extra= nn.Sequential(
                nn.Conv2d(ch_in,ch_out*ResBlk.expansion,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out*ResBlk.expansion)
            )

    def forward(self,x):
        """
        :param x: [b,ch,h,w]
        :return:
        """
        # shortcut
        # element-wise add :[b,ch_in,h,w] + [b,ch_out,h,w]
        return F.relu(self.blockfunc(x)+self.extra(x))

class ResNet(nn.Module):
    def __init__(self,a,b,c,d):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.Convblock1 = nn.Sequential()
        for i in range(1,a+1):
            self.Convblock1.add_module(name="Convblock1 {}th".format(i),module=ResBlk(ch_in = self.in_channels, ch_out = 64))

        self.Convblock2 = nn.Sequential()
        for i in range(1,b+1):
            if i==1:
                self.Convblock2.add_module(name="Convblock2 {}th".format(i), module=ResBlk(ch_in = self.in_channels, ch_out = 128,stride=2))
                self.in_channels = 128
            else:
                self.Convblock2.add_module(name="Convblock2 {}th".format(i), module=ResBlk(ch_in = self.in_channels, ch_out = 128))

        self.Convblock3 = nn.Sequential()
        for i in range(1,c+1):
            if i==1:
                self.Convblock3.add_module(name="Convblock3 {}th".format(i), module=ResBlk(ch_in = self.in_channels, ch_out = 256,stride=2))
                self.in_channels = 256
            else:
                self.Convblock3.add_module(name="Convblock3 {}th".format(i), module=ResBlk(ch_in = self.in_channels, ch_out = 256))

        self.Convblock4 = nn.Sequential()
        for i in range(1, d + 1):
            if i==1:
                self.Convblock4.add_module(name="Convblock4 1th", module=ResBlk(ch_in = self.in_channels, ch_out = 512,stride=2))
                self.in_channels = 512
            else:
                self.Convblock4.add_module(name="Convblock4 2th", module=ResBlk(ch_in = self.in_channels, ch_out = 512))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fallen = nn.Flatten()
        self.fc = nn.Linear(25088,10)

    def forward(self,x):
        out = self.conv1(x)
        # print("layer1:", out.shape)
        out = self.Convblock1(out)
        # print("layer2:", out.shape)
        out = self.Convblock2(out)
        # print("layer3:", out.shape)
        out = self.Convblock3(out)
        # print("layer4:", out.shape)
        out = self.Convblock4(out)
        # print("layer5:", out.shape)
        out = self.fallen(out)
        out = self.fc(out)
        # print("layer6:", out.shape)
        return out

def ResNet34():
    return ResNet(3, 4, 6, 3)

def ResNet18():
    return ResNet(2, 2, 2, 2)

def main():
    tmp = torch.rand(2,3,227,227)
    model = ResNet34()
    out = model(tmp)
    print(out.shape)


if __name__ == '__main__':
    main()