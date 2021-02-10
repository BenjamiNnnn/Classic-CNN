import torch
from torch import nn

class DenseBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(DenseBlock, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=ch_in,out_channels=ch_out,kernel_size=1,padding=1)
        self.layer2 = nn.Conv2d(in_channels=ch_in,out_channels=ch_out,kernel_size=3)
    def forward(self,x):
        return torch.cat([x,self.layer2(self.layer1(x))],dim=1)

class DenseNet121(nn.Module):
    growth_rate = 32
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=DenseNet121.growth_rate,kernel_size=7,stride=2),
            nn.MaxPool2d(kernel_size=3,stride=2))

        self.layer2 = nn.Sequential()
        for i in range(6):
            self.layer2.add_module(name="DenseBlock1 {}th".format(i),module=DenseBlock(ch_in=DenseNet121.growth_rate,ch_out=DenseNet121.growth_rate))
            DenseNet121.growth_rate = 2 * DenseNet121.growth_rate

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=DenseNet121.growth_rate,out_channels=DenseNet121.growth_rate,kernel_size=1),
            nn.AvgPool2d(kernel_size=2,stride=2))

        self.layer4 = nn.Sequential()
        for i in range(12):
            self.layer4.add_module(name="DenseBlock2 {}th".format(i), module=DenseBlock(ch_in=DenseNet121.growth_rate, ch_out=DenseNet121.growth_rate))
            DenseNet121.growth_rate = 2 * DenseNet121.growth_rate

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=DenseNet121.growth_rate, out_channels=DenseNet121.growth_rate, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))

        self.layer6 = nn.Sequential()
        for i in range(24):
            self.layer6.add_module(name="DenseBlock3 {}th".format(i), module=DenseBlock(ch_in=DenseNet121.growth_rate, ch_out=DenseNet121.growth_rate))
            DenseNet121.growth_rate = 2 * DenseNet121.growth_rate

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=DenseNet121.growth_rate, out_channels=DenseNet121.growth_rate, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))

        self.layer8 = nn.Sequential()
        for i in range(16):
            self.layer8.add_module(name="DenseBlock4 {}th".format(i), module=DenseBlock(ch_in=DenseNet121.growth_rate, ch_out=DenseNet121.growth_rate))
            DenseNet121.growth_rate = 2 * DenseNet121.growth_rate

        self.layer9 = nn.AvgPool2d(kernel_size=7)
        self.flaten = nn.Flatten()
        self.layer10 = nn.Linear(20,10)

    def forward(self,x):
        out = self.layer1(x)
        print("layer1:", out.shape)
        out = self.layer2(out)
        print("layer2:", out.shape)
        out = self.layer3(out)
        print("layer3:", out.shape)
        out = self.layer4(out)
        print("layer4:", out.shape)
        out = self.layer5(out)
        print("layer5:", out.shape)
        out = self.layer6(out)
        print("layer6:", out.shape)
        out = self.layer7(out)
        print("layer7:", out.shape)
        out = self.layer8(out)
        print("layer8:", out.shape)
        out = self.layer9(out)
        print("layer9:", out.shape)
        out = self.flaten(out)
        out = self.layer10(out)
        print("layer10:", out.shape)
        return out

def main():
    tmp = torch.rand(2, 3, 227, 227)
    model = DenseNet121()
    out = model(tmp)
    print(out.shape)

if __name__ == '__main__':
    main()