import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, ReLU, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras import Model

class Conv2DWithBN(Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        """
        A custom combination layer that consists of a convolutional filter, with ReLU, followed by
        batch normalization.

        Input
        - filters: number of filters
        - kernel_size: integer or tuple of integers that dictate the size of the filters (kernel_size by kernel_size)
        - strides: integer or tuple of ints that dictate the # of pixels to shift each filter over the input
        - padding: one of two strings - 'valid' or 'same' - where 'valid' means no padding and 'same' keeps
        dimensionality

        """
        super(Conv2DWithBN, self).__init__()
        self.conv = Conv2D(filters, kernel_size, strides, padding=padding, activation='relu')
        self.bn = BatchNormalization()

    def call(self, input, training):
        """
        The forward function of the custom layer that applies a conv filter, ReLU, and batch normalization.

        Input:
        - input: a tensor of shape (B x H x W x C).
        - training: boolean that determines what mode of batch normalization will be used.

        Returns:
        - x: the output after applying all individual layers.
        """
        x = self.conv(input)
        x = self.bn(x, training)
        return x
        

class ResNet(Model):
    def __init__(self, num_classes, name, repeats=[3,4,6,3]):
        """
        A modular ResNet model based on different combinations of custom block layers. Models follow the same basic 
        structure, starting with a 7x7 conv layer and 3x3 max pool, followed by skip connection blocks repeated
        various amounts of times depending on the depth of the model, wrapped up by global average pooling and a
        final dense layer mapping all filters to output classes (with softmax activation).

        Input:
        - num_classes: number of classification classes.
        - name: type of resnet model
        - repeats: list of ints that dictates how many blocks of each filter size [64, 128, 256, 512] will be 
        added to the model.
        """
        super(ResNet, self).__init__(name=name)
        self.repeats = repeats

        self.all_layers = {}
        # setup of the starting universal starting layers
        self.all_layers['conv_1'] = Conv2DWithBN(filters=64, kernel_size=7, strides=2, padding='same')
        self.all_layers['max_pool'] = MaxPooling2D(pool_size=3, strides=2, padding='same')

        # specific setup of blocks using repeats to decide how many times each filter size is repeated
        filter_sizes = [64, 128, 256, 512]
        for i, repeat in enumerate(repeats):
            for j in range(repeat):
                # dictates whether or now H and W will be halved
                curr_stride = 2
                if j != 0 or i == 0:
                    curr_stride = 1
                    
                self.all_layers[f'conv_{i+2}_{j+1}'] = Block(filter_size=filter_sizes[i], stride=curr_stride, name=(f'conv_{i+2}_{j+1}'))
        
        # final layers to map to output classes
        self.all_layers['global_avg_pool'] = GlobalAveragePooling2D()
        self.all_layers['fully_connected'] = Dense(units=num_classes, activation='softmax')

    def call(self, input, training):
        """
        The forward function that updates input after all conv filters, 
        pooling, and fully connected layers. 

        Input:
        - input: tensor with shape (B x H x W x C).
        - training: boolean that dictates whether batch normalization should update or use running parameters.

        Returns:
        - x: a tensor of shape (B x C) where C = number of classes and each ith value is the probability that 
        the input was of class i
        """

        # First universal layers
        x = self.all_layers['conv_1'](input, training)
        x = self.all_layers['max_pool'](x)

        # All blocks
        for i, repeat in enumerate(self.repeats):
            for j in range(repeat):
                x = self.all_layers[f'conv_{i+2}_{j+1}'](x, training)

        # Final mapping to classes
        x = self.all_layers['global_avg_pool'](x)
        x = self.all_layers['fully_connected'](x)
        return x


class Block(Layer):
    def __init__(self, filter_size, stride, name):
        """
        A custom layer setup to represent blocks from ResNet50, ResNet101, and ResNet152 architectures.
        Each block consists of a 1x1 conv -> 3x3 conv -> 1x1 conv with a skip connection addition at the end.
        There is a final transformation layer that can be learned if dimensions between skip connections
        do not match.

        Input:
        - filter_size: initial filter size, output channels = filter_size * 4.
        - stride: The number of pixels between adjacent receptive fields
        in the horizontal and vertical directions.
        - name: The name of the block (ex: 'conv_2_3').

        """
        super(Block, self).__init__(name=name)
        self.fs = filter_size

        # init dictionary to hold all block layers
        self.layers = {}

        # setup of conv layers
        self.layers['conv_1'] = Conv2DWithBN(filters=filter_size, kernel_size=1, strides=1, padding='valid')
        self.layers['conv_2'] = Conv2DWithBN(filters=filter_size, kernel_size=3, strides=stride, padding='same')
        self.layers['conv_3'] = Conv2DWithBN(filters=(filter_size * 4), kernel_size=1, strides=1, padding='valid')

        # set up of activation and transformation filter
        self.layers['relu'] = ReLU()
        self.layers['transform'] = Conv2DWithBN(filters=(filter_size * 4), kernel_size=1, strides=stride, padding='same')

        # if H and W dimensionality will be reduced by stride, accounts by transforming 
        # input to match the output shape
        if stride == 1: self.dotted = False
        else: self.dotted = True     

    def call(self, input, training):
        """
        The forward function that updates input by sending it through 3 convolutional filters (each followed by 
        ReLU activation and batch normalization). The output of the convolutional filters is then added with the
        starting input (shape adjusted).

        Input:
        - input: tensor with shape (B x H x W x C).
        - training: boolean that dictates whether batch normalization should update or use running parameters.

        Returns:
        - x: a tensor after applying ReLU.
        """

        # if num_channels of the input is not the same size as the output shape, transform input
        if input.shape[3] != self.fs * 4:
            self.dotted = True

        # Handles all 3 convolutional filters
        x = input
        for i in range(3):
            x = self.layers[f'conv_{i+1}'](x, training)

        # Handles matching sizes between input and output
        if self.dotted:
            y = self.layers['transform'](input, training)
            x = tf.add(x, y)
        else:
            x = tf.add(x, input)
        
        return self.layers['relu'](x)
    


def ResNet50(num_classes):
    """
    Creates an instance of a ResNet50 model.

    Input:
    - num_classes: number of classes in dataset

    Returns:
    - an instace of ResNet50
    """
    return ResNet(num_classes=num_classes, name='ResNet50', repeats=[3,4,6,3])

def ResNet101(num_classes):
    """
    Creates an instance of a ResNet101 model.

    Input:
    - num_classes: number of classes in dataset

    Returns:
    - an instace of ResNet101
    """
    return ResNet(num_classes=num_classes, name='ResNet101', repeats=[3,4,23,3])

def ResNet152(num_classes):
    """
    Creates an instance of a ResNet152 model.

    Input:
    - num_classes: number of classes in dataset

    Returns:
    - an instace of ResNet152
    """
    return ResNet(num_classes=num_classes, name='ResNet152', repeats=[3,8,36,3])
