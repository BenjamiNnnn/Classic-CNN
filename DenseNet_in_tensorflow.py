import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Layer, BatchNormalization, ReLU, AvgPool2D, Dense
from tensorflow.keras.models import Model, Sequential


class ConvBlock(Layer):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.BN1 = BatchNormalization()
        self.Activation1 = ReLU()
        self.Conv1 = Conv2D(filters=32 * 4, kernel_size=[1, 1], padding='same')

        self.BN2 = BatchNormalization()
        self.Activation2 = ReLU()
        self.Conv2 = Conv2D(filters=32, kernel_size=[3, 3], padding='same')

        self.BASCD1 = Sequential([self.BN1, self.Activation1, self.Conv1])
        self.BASCD2 = Sequential([self.BN2, self.Activation2, self.Conv2])

    def call(self, inputs):
        out = self.BASCD1(inputs)
        # print('ConvBlock out1 shape:',out.shape)
        out = self.BASCD2(out)
        # print('ConvBlock out2 shape:',out.shape)
        return out

class DenseBlock(Layer):
    def __init__(self, numberLayers):
        super(DenseBlock, self).__init__()
        self.num = numberLayers
        self.ListDenseBlock = []
        for i in range(self.num):
            self.ListDenseBlock.append(ConvBlock())

    def call(self, inputs):
        data = inputs
        for i in range(self.num):
            # print('data', i, 'shape:', data.shape)
            out = Sequential(self.ListDenseBlock[i])(data)
            # print('out',i,'shape:',out.shape)
            data = tf.concat([out, data], axis=3)

        return data


class TransLayer(Layer):
    def __init__(self):
        super(TransLayer, self).__init__()
        self.BN1 = BatchNormalization()
        self.Conv1 = Conv2D(64, kernel_size=[1, 1])
        self.pool1 = AvgPool2D(pool_size=[2, 2], strides=2)

    def call(self, inputs):
        out = self.BN1(inputs)
        out = self.Conv1(out)
        out = self.pool1(out)
        return out


class DenseNet(Model):
    def __init__(self):
        super(DenseNet, self).__init__()
        # 卷积层
        self.Conv = Conv2D(filters=32, kernel_size=(5, 5), strides=2)
        self.Pool = MaxPool2D(pool_size=(2, 2), strides=2)

        # DenseBlock 1
        self.layer1 = DenseBlock(6)
        # Transfer 1
        self.TransLayer1 = TransLayer()

        # DenseBlock 2
        self.layer2 = DenseBlock(12)
        # Transfer 2
        self.TransLayer2 = TransLayer()

        # DenseBlock 3
        self.layer3 = DenseBlock(24)
        # Transfer 3
        self.TransLayer3 = TransLayer()

        # DenseBlock 4
        self.layer4 = DenseBlock(16)
        # Transfer 4
        self.TransLayer4 = AvgPool2D(pool_size=(7, 7))

        self.softmax = Dense(3)

    def call(self, inputs):
        out = self.Conv(inputs)
        # print('Conv shape:',out.shape)    # (None, 112, 112, 32)
        out = self.Pool(out)
        # print('Pool shape:',out.shape)    # (None, 56, 56, 32)
        out = self.layer1(out)
        # print('layer1 shape:', out.shape)   # (None, 56, 56, 224)
        out = self.TransLayer1(out)
        # print('TransLayer1 shape:', out.shape)  # (None, 28, 28, 64)

        out = self.layer2(out)
        # print('layer2 shape:', out.shape)   # (None, 28, 28, 448)
        out = self.TransLayer2(out)
        # print('TransLayer2 shape:', out.shape)  # (None, 14, 14, 64)

        out = self.layer3(out)
        # print('layer3 shape:', out.shape)   # (None, 14, 14, 832)
        out = self.TransLayer3(out)
        # print('TransLayer3 shape:', out.shape)  # (None, 7, 7, 64)

        out = self.layer4(out)
        # print('layer4 shape:', out.shape)   # (None, 7, 7, 576)
        out = self.TransLayer4(out)
        # print('TransLayer4 shape:', out.shape)  # (None, 1, 1, 576)

        out = self.softmax(out)
        return out



def main():
    DenseNet121 = DenseNet()
    DenseNet121.build(input_shape=(None, 227, 227, 3))
    tmp = tf.random.normal([3, 227, 227, 3])
    out = DenseNet121(tmp)
    print(out.shape)


if __name__ == '__main__':
    main()