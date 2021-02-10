import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.models import Model

# 定义AlexNet
class self_AlexNet(Model):
    def __init__(self):
        super(self_AlexNet,self).__init__()
        self.conv1 = Conv2D(filters=96,kernel_size=(11,11),strides=4,activation=relu)
        self.pool1 = MaxPool2D(pool_size=(3,3),strides=2)

        self.conv2 = Conv2D(filters=256,kernel_size=(5,5),strides=1,padding='same',activation=relu)
        self.pool2 = MaxPool2D(pool_size=(3,3),strides=2)

        self.conv3 = Conv2D(filters=384,kernel_size=(3,3),padding='same',activation=relu)

        self.conv4 = Conv2D(filters=384,kernel_size=(3,3),padding='same',activation=relu)

        self.conv5 = Conv2D(filters=256,kernel_size=(3,3),padding='same',activation=relu)
        self.pool5 = MaxPool2D(pool_size=(3,3),strides=2)

        self.fallten = Flatten()

        self.fc6 = Dense(units=2048,activation=relu)
        self.drop6 = Dropout(rate=0.5)

        self.fc7 = Dense(units=2048,activation=relu)
        self.drop7 = Dropout(rate=0.5)

        self.fc8 = Dense(units=3,activation=softmax)

    def call(self,input):
        out = self.conv1(input)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.pool5(out)
        # 原论文是使用两块GPU并行计算的 所以是4096
        # 这里应该修改为9216
        # out = tf.reshape(out,[-1,9216])
        out = self.fallten(out)

        out = self.fc6(out)
        out = self.drop6(out)
        out = self.fc7(out)
        out = self.drop7(out)
        out = self.fc8(out)
        return out



def main():
    AlexNet = self_AlexNet()
    AlexNet.build(input_shape=(None, 227, 227, 3))
    tmp = tf.random.normal([3, 227, 227, 3])
    out = AlexNet(tmp)
    print(out.shape)




if __name__ == '__main__':
    main()