from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, concatenate, UpSampling2D, MaxPool2D

class conv_block(Model):
    def __init__(self, n_filters, is_batchnorm=False):
        super(conv_block, self).__init__()
    
        self.conv1 = Conv2D(n_filters, 3, activation='relu', padding = 'same')
        self.conv2 = Conv2D(n_filters, 3, activation='relu', padding = 'same')

    def call(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        return out

class up_block(Model):
    def __init__(self, n_filters):
        super(up_block, self).__init__()
        self.up = UpSampling2D(size=(2, 2))
        self.conv = conv_block(n_filters)


    def call(self, x1, x2):
        out2 = self.up(x2)
        out = self.conv(concatenate([x1, out2]))
        return out


class unet(Model):
    def __init__(self, n_channels=3, n_classes=12, feature_scale=32, is_deconv=True, is_batchnorm=True):
        super(unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.filters = [64, 128, 256, 512, 1024]
        self.filters = [int(x/self.feature_scale) for x in self.filters]
        
        self.conv1 = conv_block(self.filters[0])
        self.conv2 = conv_block(self.filters[1])
        self.conv3 = conv_block(self.filters[2])
        self.conv4 = conv_block(self.filters[3])

        self.center = conv_block(self.filters[4])

        self.up4 = up_block(self.filters[3])
        self.up3 = up_block(self.filters[2])
        self.up2 = up_block(self.filters[1])
        self.up1 = up_block(self.filters[0])

        self.out = Conv2D(n_classes, 1, activation='relu')

    def call(self, x):
        conv1 = self.conv1(x)
        maxpool1 = MaxPool2D()(conv1)
        
        conv2 = self.conv2(maxpool1)
        maxpool2 = MaxPool2D()(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = MaxPool2D()(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = MaxPool2D()(conv4)

        center = self.center(maxpool4)

        up4 = self.up4(conv4, center)
        up3 = self.up3(conv3, up4)
        up2 = self.up2(conv2, up3)
        up1 = self.up1(conv1, up2)

        out = self.out(up1)

        return out
