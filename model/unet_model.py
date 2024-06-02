#!/usr/bin/python3

import torch
import torch.nn as nn

# Reference Soruce : https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py


class UNET(nn.Module):

    def __init__(
        self, in_channel=1, pooling='max', hiddenactivation='GELU', encoder_ksize=(3, 3), decoder_ksize=(3, 3), device=None, outputActivate=None,
        initial_feature_map_size=64):
        super().__init__()
        self.device = None
        self.in_channel = in_channel
        self.hiddenactivation = hiddenactivation
        if (device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.poolinglayer = nn.MaxPool2d((2, 2)) if (pooling == 'max') else nn.AvgPool2d((2, 2))
        self.initial_feature_map_size = initial_feature_map_size
        
        # Calculate padding based on input kernel size
        self.encoder_ksize = encoder_ksize
        self.decoder_ksize = decoder_ksize
        self.encoder_padding = self.calculate_padding(encoder_ksize)
        self.decoder_padding = self.calculate_padding(decoder_ksize)
        
        self.unet_encoder1 = self._encoder_convBlock(
            self.in_channel, self.initial_feature_map_size, self.encoder_ksize, self.encoder_padding, 
            hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder2 = self._encoder_convBlock(
            self.initial_feature_map_size, self.initial_feature_map_size * 2, self.encoder_ksize, self.encoder_padding, 
            hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder3 = self._encoder_convBlock(
            self.initial_feature_map_size * 2, self.initial_feature_map_size * 4, self.encoder_ksize, self.encoder_padding, 
            hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder4 = self._encoder_convBlock(
            self.initial_feature_map_size * 4, self.initial_feature_map_size * 8, self.encoder_ksize, self.encoder_padding, 
            hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder5 = self._encoder_convBlock(
            self.initial_feature_map_size * 8, self.initial_feature_map_size * 16, self.encoder_ksize, self.encoder_padding, 
            hiddenactivation=self.hiddenactivation).to(self.device)

        self.unet_decoder1 = self._decoder_convBlock(
            self.initial_feature_map_size * 16, self.initial_feature_map_size * 8, self.decoder_ksize, self.decoder_padding, 
            hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_decoder2 = self._decoder_convBlock(
            self.initial_feature_map_size * 8, self.initial_feature_map_size * 4, self.decoder_ksize, self.decoder_padding, 
            hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_decoder3 = self._decoder_convBlock(
            self.initial_feature_map_size * 4, self.initial_feature_map_size * 2, self.decoder_ksize, self.decoder_padding, 
            hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_decoder4 = self._decoder_convBlock(
            self.initial_feature_map_size * 2, self.initial_feature_map_size, self.decoder_ksize, self.decoder_padding, 
            hiddenactivation=self.hiddenactivation).to(self.device)
        self.output = nn.Conv2d(self.initial_feature_map_size, self.in_channel, kernel_size=1, padding=0).to(self.device)

    # Calculate padding to maintain spatial dimensions
    def calculate_padding(self, kernel_size):
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        return padding

    def forward(self, x):
        x = x.to(self.device)
        encoder1 = self.unet_encoder1(x)
        poolLayer = self.poolinglayer(encoder1)
        encoder2 = self.unet_encoder2(poolLayer)
        poolLayer = self.poolinglayer(encoder2)
        encoder3 = self.unet_encoder3(poolLayer)
        poolLayer = self.poolinglayer(encoder3)
        encoder4 = self.unet_encoder4(poolLayer)
        poolLayer = self.poolinglayer(encoder4)
        encoder5 = self.unet_encoder5(poolLayer)

        upscale = self.unet_decoder1(encoder5, encoder4)
        upscale = self.unet_decoder2(upscale, encoder3)
        upscale = self.unet_decoder3(upscale, encoder2)
        upscale = self.unet_decoder4(upscale, encoder1)
        upscale = self.output(upscale)
        return upscale

    class _encoder_convBlock(nn.Module):

        def __init__(self, in_channel, out_channel, ksize=(3, 3), padding=1 , dropout_p=0, hiddenactivation='GELU'):
            super().__init__()
            hiddenactivation = getattr(nn, hiddenactivation)
            activation = hiddenactivation()
            if (dropout_p > 0):
                self.conv2d = nn.Sequential(
                    nn.Conv2d(in_channel, _channel, kernel_size=ksize, padding=padding),
                    nn.BatchNorm2d(int_channel),
                    nn.Dropout2d(dropout_p),
                    nn.Conv2d(out_channel, out_channel, kernel_size=ksize, padding=padding),
                    nn.BatchNorm2d(out_channel),
                    nn.Dropout2d(dropout_p),
                    activation
                )
            else:
                self.conv2d = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=padding),
                    #nn.BatchNorm2d(in_channel),
                    activation,
                    nn.Conv2d(out_channel, out_channel, kernel_size=ksize, padding=padding),
                    #nn.BatchNorm2d(out_channel),
                    activation
                )

        def forward(self, x):
            return self.conv2d(x)

    class _decoder_convBlock(nn.Module):

        def __init__(self, in_channel, out_channel, ksize=(3,3), padding=1, dropout_p=0, hiddenactivation='GELU'):
            super().__init__()
            hiddenactivation = getattr(nn,hiddenactivation)
            activation =  hiddenactivation()
            if (dropout_p > 0):
                self.upscale = nn.Sequential(
                    nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(2, 2), stride=2),
                    nn.BatchNorm2d(out_channel),
                    nn.Dropout2d(dropout_p),
                )
                self.conv2d = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=ksize, padding=padding),
                    nn.BatchNorm2d(in_channel),
                    nn.Dropout2d(dropout_p),
                    nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=padding),
                    nn.BatchNorm2d(out_channel),
                    nn.Dropout2d(dropout_p),
                    activation
                )
            else:
                self.upscale = nn.Sequential(
                    nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(2, 2), stride=2),
                    nn.BatchNorm2d(out_channel),
                )
                self.conv2d = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=ksize, padding=padding),
                    #nn.BatchNorm2d(in_channel),
                    activation,
                    nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=padding),
                    #nn.BatchNorm2d(out_channel),
                    activation
                )

        def forward(self, x, y):
            x = self.upscale(x)
            x = self.conv2d(torch.cat([x, y], 1))
            return x

