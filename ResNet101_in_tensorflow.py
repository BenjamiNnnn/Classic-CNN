import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Layer,BatchNormalization,ReLU,AvgPool2D
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model,Sequential

class BasicBlock(Layer):

    def __init__(self,filter_num1,filter_num2,stride):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(filters=filter_num1,kernel_size=[1,1],strides=stride)
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()

        self.conv2 = Conv2D(filters=filter_num1,kernel_size=[3,3],strides=stride,padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()

        self.conv3 = Conv2D(filters=filter_num2,kernel_size=[1,1],strides=stride,padding='same')
        self.bn3 = BatchNormalization()

        self.downsample = Sequential([Conv2D(filters=filter_num2,kernel_size=[1,1],strides=stride),
                                      BatchNormalization(),
                                      ReLU()])

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 计算残差
        residual = self.downsample(inputs)
        out += residual
        out = relu(out)

        return out

class self_ResNet101(Model):
    def __init__(self):
        super(self_ResNet101,self).__init__()
        self.conv1 = Sequential([Conv2D(filters=64,kernel_size=[7,7],strides=2,padding='same'),
                                BatchNormalization(),
                                ReLU(),
                                MaxPool2D(pool_size=[3,3],strides=2)])

        self.layerN = []
        # layer1
        for i in range(3):
            self.layerN.append(BasicBlock(32,64,1))
        # layer2
        for i in range(4):
            self.layerN.append(BasicBlock(64,128,1))
        # layer3
        for i in range(23):
            self.layerN.append(BasicBlock(128,256,1))
        # layer4
        for i in range(3):
            self.layerN.append(BasicBlock(256,512,1))

        self.layerN = Sequential(self.layerN)
        self.Avg = AvgPool2D(pool_size=[7,7],strides=1)
        self.flatten = Flatten()
        self.fc = Dense(units=3)


    def call(self,data):
        out = self.conv1(data)
        out = self.layerN(out)
        out = self.Avg(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


def main():
    ResNet101 = self_ResNet101()
    ResNet101.build(input_shape=(None, 227, 227, 3))
    tmp = tf.random.normal([3, 227, 227, 3])
    out = ResNet101(tmp)
    print(out.shape)

if __name__ == '__main__':
    main()