import torch
from torch import nn

class AlexNet8(nn.Module):
    """
    input: 227*227
    """
    def __init__(self):
        super(AlexNet8, self).__init__()
        # 第一层 卷积核大小(11*11*96) 卷积步长(4) 最大值池化(3*3) 池化步长(2)
        # [227,227,3] => [27,27,96]
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3,stride=2))
        # 第二层 卷积核大小(5*5*256) 卷积步长(1) 填充(2) 最大值池化(3*3) 池化步长(2)
        # [27,27,96] => [13,13,256]
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=1),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2))
        # 第三层 卷积核大小(3*3*384) 卷积步长(1)  填充(1)
        # [13,13,256] => [13,13,384]
        self.layer3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1, stride=1)
        # 第四层 卷积核大小(3*3*384) 卷积步长(1)  填充(1)
        # [13,13,384] => [13,13,384]
        self.layer4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, stride=1)
        # 第五层 卷积核大小(3*3*384) 卷积步长(2)  填充(1) 最大值池化(3*3) 池化步长(2)
        # [13,13,384] => [6,6,256]
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2))
        # [6,6,256] => [6*6*256]
        self.fallen = nn.Flatten()
        # 第六层 全连接层
        self.layer6 = nn.Linear(9216,4096)
        # 第七层 全连接层
        self.layer7 = nn.Linear(4096,4096)
        # 第八层 全连接层
        self.layer8 = nn.Linear(4096,10)

    def forward(self,x):
        out = self.layer1(x)
        print("layer1:",out.shape)
        out = self.layer2(out)
        print("layer2:",out.shape)
        out = self.layer3(out)
        print("layer3:",out.shape)
        out = self.layer4(out)
        print("layer4:",out.shape)
        out = self.layer5(out)
        print("layer5:",out.shape)
        out = self.fallen(out)
        out = self.layer6(out)
        print("layer6:",out.shape)
        out = self.layer7(out)
        print("layer7:",out.shape)
        out = self.layer8(out)
        print("layer8:",out.shape)
        return out

def main():
    tmp = torch.rand(27,3,227,227)
    model = AlexNet8()
    out = model(tmp)
    print(out.shape)

if __name__ == '__main__':
    main()