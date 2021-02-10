import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

# 定义VGG16
class self_VGG16(Model):
    def __init__(self):
        super(self_VGG16,self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=relu)
        self.conv2 = Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=relu)
        self.pool1 = MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
        # unit 2
        self.conv3 = Conv2D(128, kernel_size=[3, 3], padding='same', activation=relu)
        self.conv4 = Conv2D(128, kernel_size=[3, 3], padding='same', activation=relu)
        self.pool2 = MaxPool2D(pool_size=[2, 2], padding='same')
        # unit 3
        self.conv5 = Conv2D(256, kernel_size=[3, 3], padding='same', activation=relu)
        self.conv6 = Conv2D(256, kernel_size=[3, 3], padding='same', activation=relu)
        self.conv7 = Conv2D(256, kernel_size=[1, 1], padding='same', activation=relu)
        self.pool3 = MaxPool2D(pool_size=[2, 2], padding='same')
        # unit 4
        self.conv8 = Conv2D(512, kernel_size=[3, 3], padding='same', activation=relu)
        self.conv9 = Conv2D(512, kernel_size=[3, 3], padding='same', activation=relu)
        self.conv10 = Conv2D(512, kernel_size=[1, 1], padding='same', activation=relu)
        self.pool4 = MaxPool2D(pool_size=[2, 2], padding='same')
        # unit 5
        self.conv11 = Conv2D(512, kernel_size=[3, 3], padding='same', activation=relu)
        self.conv12 = Conv2D(512, kernel_size=[3, 3], padding='same', activation=relu)
        self.conv13 = Conv2D(512, kernel_size=[1, 1], padding='same', activation=relu)
        self.pool5 = MaxPool2D(pool_size=[2, 2], padding='same')
        # 全连接
        self.fc14 = Dense(4096, activation=relu)
        self.fc15 = Dense(4096, activation=relu)
        self.fc16 = Dense(1000, activation=None)

    def call(self,input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.pool1(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool2(out)

        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.pool3(out)

        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.pool4(out)

        out = self.conv11(out)
        out = self.conv12(out)
        out = self.conv13(out)
        out = self.pool5(out)

        out = Flatten()(out)

        out = self.fc14(out)
        out = self.fc15(out)
        out = self.fc16(out)

        return out



def main():
    VGG16 = self_VGG16()
    VGG16.build(input_shape=(None, 227, 227, 3))
    tmp = tf.random.normal([3, 227, 227, 3])
    out = VGG16(tmp)
    print(out.shape)



if __name__ == '__main__':
    main()