import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, ReLU, BatchNormaliztion, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

class Conv2DWithBN(Layer):
    def __init__(self, filters,
                kernel_size,
                strides,
                padding):
        self.conv = Conv2D(filters, kernel_size, strides, padding=padding, activation='relu')
        self.bn = BatchNormaliztion()

    def call(self, input):
        x = self.conv(input)
        x = self.bn(x)
        return x
        

class ResNet(Model):
    def __init__(self, img_shape, num_classes, repeats=[3,4,6,3]):
        super(ResNet, self).__init__()

        # Expect (X, Y, C)
        X, Y, C = img_shape
        if X == 224 and Y == X:
            padding = 3
        elif X < 224 and Y == X:
            padding = (tf.abs(224 - X) + 7) // 2
        else:
            print(f'IMAGE SHAPE: {img_shape}, expecting dimensions (224, 224, 3)')
            return
        
        self.layers = {}
        self.layers['conv_1'] = Conv2DWithBN(filters=64, kernel_size=7, strides=2, padding=padding)
        self.layers['max_pool'] = MaxPooling2D(pool_size=3, strides=2)

        for i, repeat in enumerate(repeats):
            for j in range(repeat):
                curr_stride = 2
                if j != 0 or i == 0:
                    curr_stride = 1
                self.layers[f'conv_{i+1}_{j+1}'] = Block(filter_size=64, stride=curr_stride)


class Block(Layer):
    def __init__(self, filter_size, stride):
        super(Block, self).__init__()
        self.layers = {}
        self.layers['conv_1'] = Conv2DWithBN(filters=filter_size, kernel_size=1, strides=1, padding=0)
        self.layers['conv_2'] = Conv2DWithBN(filters=filter_size, kernel_size=3, strides=stride, padding=1)
        self.layers['conv_3'] = Conv2DWithBN(filters=(filter_size * 4), kernel_size=1, strides=1, padding=0)
        self.layers['relu'] = ReLU()
        self.layers['transform'] = Conv2DWithBN(filters=(filter_size * 4), kernel_size=1, strides=stride, padding=1)
        if stride == 1: self.dotted = False
        else: self.dotted = True
        

    def call(self, input):
        x = input
        for i in range(1,3):
            x = self.layers[f'conv_{i}'](x)
        if self.dotted:
            x = tf.add(x, self.layers['transform'](input))
        else:
            x = tf.add(x, input)
        return self.layers['relu'](x)