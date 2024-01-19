import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, ReLU, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

class Conv2DWithBN(Layer):
    def __init__(self, filters,
                kernel_size,
                strides,
                padding):
        super(Conv2DWithBN, self).__init__()
        self.conv = Conv2D(filters, kernel_size, strides, padding=padding, activation='relu')
        self.bn = BatchNormalization()

    def call(self, input, training):
        x = self.conv(input)
        x = self.bn(x, training)
        return x
        

class ResNet(Model):
    def __init__(self, img_shape, num_classes, repeats=[3,4,6,3]):
        super(ResNet, self).__init__(name='ResNet')
        self.repeats = repeats
        # Expect (X, Y, C)
        # X, Y, C = img_shape
        # if X == 224 and Y == X:
        #     padding = 3
        # elif X == 200 and Y == X:
        #     # padding = (tf.abs(222 - X) + 7) // 2
        #     padding = (230 - X)/2
        # else:
        #     print(f'IMAGE SHAPE: {img_shape}, expecting dimensions (224, 224, 3)')
        #     return
        
        self.all_layers = {}
        self.all_layers['conv_1'] = Conv2DWithBN(filters=64, kernel_size=7, strides=2, padding='same')
        self.all_layers['max_pool'] = MaxPooling2D(pool_size=3, strides=2, padding='same')

        filter_sizes = [64, 128, 256, 512]
        for i, repeat in enumerate(repeats):
            for j in range(repeat):
                curr_stride = 2
                if j != 0 or i == 0:
                    curr_stride = 1
                fs = filter_sizes[i]
                if j != 0:
                    fs = filter_sizes[i] * 4
                self.all_layers[f'conv_{i+1}_{j+1}'] = Block(filter_size=fs, stride=curr_stride) # TODO: fix filter size
        
        self.all_layers['global_avg_pool'] = GlobalAveragePooling2D()
        self.all_layers['fully_connected'] = Dense(units=num_classes, activation='softmax')

    def call(self, input, training):
        x = self.all_layers['conv_1'](input, training)
        x = self.all_layers['max_pool'](x)
        for i, repeat in enumerate(self.repeats):
            for j in range(repeat):
                x = self.all_layers[f'conv_{i+1}_{j+1}'](x, training)
        x = self.all_layers['global_avg_pool'](x)
        x = self.all_layers['fully_connected'](x)
        return x

class Block(Layer):
    def __init__(self, filter_size, stride):
        super(Block, self).__init__()
        self.layers = {}
        self.layers['conv_1'] = Conv2DWithBN(filters=filter_size, kernel_size=1, strides=1, padding='valid')
        self.layers['conv_2'] = Conv2DWithBN(filters=filter_size, kernel_size=3, strides=stride, padding='same')
        self.layers['conv_3'] = Conv2DWithBN(filters=(filter_size * 4), kernel_size=1, strides=1, padding='valid')
        self.layers['relu'] = ReLU()
        self.layers['transform'] = Conv2DWithBN(filters=(filter_size * 4), kernel_size=1, strides=stride, padding='same')
        if stride == 1: self.dotted = False
        else: self.dotted = True
        

    def call(self, input, training):
        x = input
        for i in range(1,3):
            x = self.layers[f'conv_{i}'](x, training)
        if self.dotted:
            x = tf.add(x, self.layers['transform'](input, training))
        else:
            x = tf.add(x, input)
        return self.layers['relu'](x)
    
def ResNet50(img_shape, num_classes):
    return ResNet(img_shape=img_shape, num_classes=num_classes, repeats=[3,4,6,4])