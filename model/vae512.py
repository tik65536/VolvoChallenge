#!/usr/bin/python3

import torch
import torch.nn as nn

# Reference Soruce : https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py


class VAE512(nn.Module):

    def __init__(self, in_channel=1, pooling='max', hiddenactivation='GELU', bottleneckactivation='GELU', encoder_ksize=(3, 3), decoder_ksize=(3, 3),
                 encoder_padding=1, decoder_padding=1, device=None):
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
        self.unet_encoder1 = self._encoder_convBlock(self.in_channel, 64, encoder_ksize, encoder_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder2 = self._encoder_convBlock(64, 128, encoder_ksize, encoder_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder3 = self._encoder_convBlock(128, 256, encoder_ksize, encoder_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder4 = self._encoder_convBlock(256, 512, encoder_ksize, encoder_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_encoder5 = self._encoder_convBlock(512, 1024, encoder_ksize, encoder_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.mu_net = self._bottleneck(1024,1,bottleneckactivation).to(self.device)
        self.logvar_net = self._bottleneck(1024,1,bottleneckactivation).to(self.device)
        self.upscale = nn.Conv2d(1,1024,kernel_size=1,padding=0).to(self.device)
        self.unet_decoder0 = self._decoder_convBlock(1, 1024, decoder_ksize, decoder_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_decoder1 = self._decoder_convBlock(1024, 512, decoder_ksize, decoder_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_decoder2 = self._decoder_convBlock(512, 256, decoder_ksize, decoder_padding,  hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_decoder3 = self._decoder_convBlock(256, 128, decoder_ksize, decoder_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_decoder4 = self._decoder_convBlock(128, 64, decoder_ksize, decoder_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.output = nn.Conv2d(64, self.in_channel, kernel_size=1,padding=0).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        encoder = self.unet_encoder1(x)
        encoder = self.poolinglayer(encoder)
        encoder = self.unet_encoder2(encoder)
        encoder = self.poolinglayer(encoder)
        encoder = self.unet_encoder3(encoder)
        encoder = self.poolinglayer(encoder)
        encoder = self.unet_encoder4(encoder)
        encoder = self.poolinglayer(encoder)
        encoder = self.unet_encoder5(encoder)
        mu = self.mu_net(encoder)
        logvar = self.logvar_net(encoder)
        encoder = self.reparametrize(mu, logvar)
        upscale = self.upscale(encoder)
        upscale = self.unet_decoder1(upscale)
        upscale = self.unet_decoder2(upscale)
        upscale = self.unet_decoder3(upscale)
        upscale = self.unet_decoder4(upscale)
        upscale = self.output(upscale)
        return upscale, mu, logvar

    def reparametrize(self,mu,log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn_like(log_var,device=self.device)
        #z= z.type_as(mu)
        return mu + sigma*z


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
                    nn.Conv2d(in_channel, in_channel, kernel_size=ksize, padding=1),
                    nn.BatchNorm2d(in_channel),
                    nn.Dropout2d(dropout_p),
                    nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=1),
                    nn.BatchNorm2d(out_channel),
                    nn.Dropout2d(dropout_p),
                    activation
                )
            else:
                self.upscale = nn.Sequential(
                    nn.ConvTranspose2d(in_channel, in_channel, kernel_size=(2, 2), stride=2),
                )
                self.conv2d = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=ksize, padding=padding),
                    #nn.BatchNorm2d(in_channel),
                    activation,
                    nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=padding),
                    #nn.BatchNorm2d(out_channel),
                    activation
                )

        def forward(self, x):
            x = self.upscale(x)
            x = self.conv2d(x)
            return x

    class _bottleneck(nn.Module):

        def __init__(self, in_channel, dim, bottleneckactivation='GELU'):
            super().__init__()
            bottleneckactivation = getattr(nn,bottleneckactivation)
            activation =  bottleneckactivation()
            self.bottleneck = nn.Sequential(
                # nn.Linear(in_channel*dim, dim),
                nn.Conv2d(in_channel,1,kernel_size=3,padding=1),
                activation
            )

        def forward(self, x):
            x = self.bottleneck(x)
            return x


