###################################################################################################
#
# Copyright (C) 2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Auto Encoder Network
"""
import torch
from torch import nn

import ai8x


class AutoEncoder(nn.Module):
    """
    Auto Encoder Model
    """
    def __init__(self,
                 num_features=64,
                 num_classes=2,  # pylint: disable=unused-argument
                 num_channels=3,  # pylint: disable=unused-argument
                 dimensions=(512, 1),  # pylint: disable=unused-argument
                 **kwargs):
        super().__init__()
        self.num_features = num_features #128  128 has not resulted in much improvement
        self.encoder = Encoder_FCNN(num_features, **kwargs)
        self.decoder = Decoder_FCNN(num_features, **kwargs)

    def forward(self, signal):
        """
        Forward propagation
        """
        encoding = self.encoder.forward(signal)
        reconstruction = self.decoder.forward(encoding)

        return reconstruction


class Encoder_FCNN(nn.Module):
    """
    Encoder Part of the Auto Encoder Model
    """
    def __init__(self, num_features=128, **kwargs):
        super().__init__()

        self.num_features = num_features
        self.out_chanels_for_the_last_layer = int(self.num_features / 4)

        self.encode_conv1 = ai8x.FusedMaxPoolConv1dBN(in_channels=3, out_channels=32, kernel_size=3,
                                                    stride=1, padding=1, bias=True,activation='ReLU')
        self.encode_conv2 = ai8x.FusedMaxPoolConv1dBN(in_channels=32, out_channels=64, kernel_size=3,
                                                    stride=1, padding=1, bias=True,activation='ReLU')
        self.encode_conv3 = ai8x.Conv1d(in_channels=64, out_channels=64,
                                                    kernel_size=3, stride=1, padding=1,bias=True,activation='ReLU')
        self.encode_conv4 = ai8x.FusedMaxPoolConv1dBN(in_channels=64, out_channels=128, kernel_size=3,
                                                    stride=1, padding=1,bias=True,activation='ReLU')
        self.encode_conv5 = ai8x.FusedMaxPoolConv1dBN(in_channels=128, out_channels=128, kernel_size=3,
                                                    stride=1, padding=1,bias=True,activation='ReLU')
        self.encode_conv6 = ai8x.FusedMaxPoolConv1dBN(in_channels=128, out_channels=128,
                                                    kernel_size=3, stride=1, padding=1,bias=True,activation='ReLU')
        self.encode_conv7 = ai8x.FusedMaxPoolConv1dBN(in_channels=128, out_channels=64, kernel_size=3,
                                                    stride=1, padding=1,bias=True,activation='ReLU')
        self.encode_conv8 = ai8x.Conv1d(in_channels=64, out_channels=64,
                                                    kernel_size=3, stride=1, padding=1,bias=True,activation='ReLU')
        self.encode_conv9 = ai8x.FusedMaxPoolConv1dBN(in_channels=64, out_channels=self.out_chanels_for_the_last_layer,
                                                    kernel_size=3, stride=1, padding=1,bias=True,activation='None')

        self.init_convs()

    def init_convs(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
            if isinstance(c, nn.Linear):
                nn.init.xavier_uniform_(c.weight)

    def encode(self, x):
        """
        Encoding Sequence
        """
        x = self.encode_conv1(x)
        x = self.encode_conv2(x)
        x = self.encode_conv3(x)
        x = self.encode_conv4(x)
        x = self.encode_conv5(x)
        x = self.encode_conv6(x)
        x = self.encode_conv7(x)
        x = self.encode_conv8(x)
        x = self.encode_conv9(x)
        return x

    def forward(self, x):
        """
        Encoding Forward Propogation
        """
        x = self.encode(x)

        return x


class Decoder_FCNN(nn.Module):
    """
    Decoder Part of the Auto Encoder Model
    """
    def __init__(self, num_features=128, **kwargs):

        super().__init__()

        self.num_features = num_features
        self.in_chanels_for_the_first_layer = int(self.num_features / 4)

        self.decode_ConvTranspose2d1 = ai8x.ConvTranspose2d(in_channels=self.in_chanels_for_the_first_layer, out_channels=64,kernel_size=3,stride=2,bias=True, padding=1,activation='ReLU')
        self.decode_Conv2d1 = ai8x.Conv2d(in_channels=64, out_channels=128,kernel_size=3,bias=True,activation='ReLU')
        self.decode_ConvTranspose2d2 = ai8x.ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=3,stride=2,bias=True, padding=1,activation='ReLU')
        self.decode_ConvTranspose2d3 = ai8x.ConvTranspose2d(in_channels=64, out_channels=32,kernel_size=3,stride=2,bias=True, padding=1,activation='ReLU')
        self.decode_ConvTranspose2d4 = ai8x.ConvTranspose2d(in_channels=32, out_channels=16,kernel_size=3,stride=2, padding=1,bias=True,activation='ReLU')
        self.decode_conv1d1 =  ai8x.Conv1d(in_channels=16, out_channels=8,
                                                    kernel_size=3, stride=1, padding=1,bias=True,batchnorm = 'Affine',activation='ReLU')
        self.decode_conv1d2 =  ai8x.Conv1d(in_channels=8, out_channels=6,
                                                    kernel_size=3, stride=1, padding=1,bias=True)

    def decode(self, x):
        """
        Decoding Sequence
        """
        x = x.view(x.shape[0], self.in_chanels_for_the_first_layer, 2, -1)
        x=self.decode_ConvTranspose2d1(x)
        x=self.decode_Conv2d1(x)
        x=self.decode_ConvTranspose2d2(x)
        x=self.decode_ConvTranspose2d3(x)
        x=self.decode_ConvTranspose2d4(x)
        x = x.view(x.shape[0],16,-1)
        x=self.decode_conv1d1(x)
        x=self.decode_conv1d2(x)
        x = x.view(x.shape[0], 3,-1)
        return x

    def forward(self, x):
        """
        Decoding Forward Propogation
        """
        x = self.decode(x)
        return x


def ai85autoencoder(pretrained=False, **kwargs):
    """
    Returns an Auto Encoder Model
    """
    assert not pretrained
    return AutoEncoder(**kwargs)


def ai85autoencoder_16feats(pretrained=False, **kwargs):
    """
    Returns an Auto Encoder Model
    """
    assert not pretrained
    return AutoEncoder(num_features=16, **kwargs)


models = [
    {
        'name': 'ai85autoencoder',
        'min_input': 1,
        'dim': 1,
    },
    {
        'name': 'ai85autoencoder_16feats',
        'min_input': 1,
        'dim': 1,
    }
]
