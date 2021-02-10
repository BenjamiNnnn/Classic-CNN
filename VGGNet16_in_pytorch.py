import torch
from torch import nn

class VGGNet16(nn.Module):
    """
    input: 224*224
    """
    def __init__(self):
        super(VGGNet16, self).__init__()
        # 第一层 卷积核大小(64,3,3,3) 卷积步长(1) 填充(1)
        # 第二层 卷积核大小(64,3,3,64) 卷积步长(1) 填充(1)
        # 最大值池化(2*2) 池化步长(2)
        # [227,227,3] =>[112, 112, 64]
        self.layer1_2 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2))
        # 第三层 卷积核大小(128,64,3,3) 卷积步长(1) 填充(1)
        # 第四层 卷积核大小(128,128,3,64) 卷积步长(1) 填充(1)
        # 最大值池化(2*2) 池化步长(2)
        # [112, 112, 64] => [56, 56, 128]
        self.layer3_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # 第五层 卷积核大小(256,128,3,3) 卷积步长(1) 填充(1)
        # 第六层 卷积核大小(256,256,3,3) 卷积步长(1) 填充(1)
        # 第七层 卷积核大小(256,256,3,3) 卷积步长(1) 填充(1)
        # 最大值池化(2*2) 池化步长(2)
        # [56, 56, 128] => [28, 28, 256]
        self.layer5_7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # 第八层 卷积核大小(512,256,3,3) 卷积步长(1) 填充(1)
        # 第九层 卷积核大小(512,512,3,3) 卷积步长(1) 填充(1)
        # 第十层 卷积核大小(512,512,3,3) 卷积步长(1) 填充(1)
        # 最大值池化(2*2) 池化步长(2)
        # [28, 28, 256] => [14, 14, 512]
        self.layer8_10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # 第十一层 卷积核大小(512,512,3,3) 卷积步长(1) 填充(1)
        # 第十二层 卷积核大小(512,512,3,3) 卷积步长(1) 填充(1)
        # 第十三层 卷积核大小(512,512,3,3) 卷积步长(1) 填充(1)
        # 最大值池化(2*2) 池化步长(2)
        # [14, 14, 512] => [7, 7, 512]
        self.layer11_13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fallen = nn.Flatten()
        # 第十四层 全连接层
        self.layer14 = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True))
        # 第十五层 全连接层
        self.layer15 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True))
        # 第十六层 全连接层
        self.layer16 = nn.Linear(4096,10)

    def forward(self,x):
        out = self.layer1_2(x)
        print("layer1_2:",out.shape)

        out = self.layer3_4(out)
        print("layer3_4:",out.shape)

        out = self.layer5_7(out)
        print("layer5_7:",out.shape)

        out = self.layer8_10(out)
        print("layer8_10:",out.shape)

        out = self.layer11_13(out)
        print("layer11_13:", out.shape)

        out = self.fallen(out)
        out = self.layer14(out)
        print("layer14:", out.shape)
        out = self.layer15(out)
        print("layer15:", out.shape)
        out = self.layer16(out)
        print("layer16:", out.shape)
        return out

def main():
    tmp = torch.rand(27,3,224,224)
    model = VGGNet16()
    out = model(tmp)
    print(out.shape)

if __name__ == '__main__':
    main()