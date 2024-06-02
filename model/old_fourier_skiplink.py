#!/usr/bin/python3

import torch
import torch.nn as nn

# Reference Soruce : https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py


class Fourier_skiplink(nn.Module):

    def __init__(self, dim, in_channel=1, project_channel=4, hidden_channel=8, pooling='max', mode=256,
                 hiddenactivation='GELU', activation='GELU', encoder_ksize=(3,3), encoder_padding=1, device=None):
        super().__init__()
        self.device = None
        self.mode = mode
        self.dim = dim
        self.in_channel = in_channel
        self.project_channel = project_channel
        self.hiddenactivation = hiddenactivation
        self.fold = (self.dim[0]//abs(self.mode)) if (self.dim[0]//abs(self.mode)>0) else 1
        if (device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.unet_Fencoder1 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold,dim=self.dim).to(self.device)
        self.unet_Fencoder2 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold).to(self.device)
        self.unet_Fencoder3 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold).to(self.device)
        self.unet_Fencoder4 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold).to(self.device)
        self.unet_Fencoder5 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold).to(self.device)
        self.unet_Fencoder6 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, skiplink=True).to(self.device)
        self.unet_Fencoder7 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, skiplink=True).to(self.device)
        self.unet_Fencoder8 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, skiplink=True).to(self.device)
        self.unet_Fencoder9 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, mode=self.mode, fold=self.fold,
                                                       inverse=True, dim=self.dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        encoder, x1 = self.unet_Fencoder1(x)
        encoder, x2 = self.unet_Fencoder2(encoder)
        encoder, x3 = self.unet_Fencoder3(encoder)
        encoder, x4 = self.unet_Fencoder4(encoder)
        encoder, _ = self.unet_Fencoder5(encoder)
        encoder = self.unet_Fencoder6(encoder, x4)
        encoder = self.unet_Fencoder7(encoder, x3)
        encoder = self.unet_Fencoder8(encoder, x2)
        encoder = self.unet_Fencoder9(encoder, x1)

        return encoder

    class _Fencoder_convBlock(nn.Module):

        def __init__(self, in_channel, out_channel, hidden_channel, ksize=(3, 3), padding=1 ,
                     hiddenactivation='GELU', activation='GELU', mode=0, fold=1, inverse=False, dim=None, skiplink=False):
            super().__init__()
            self.mode=mode
            self.fold = fold
            hiddenactivation = getattr(nn, hiddenactivation)
            activation = getattr(nn, activation)
            self.hidden_activation = hiddenactivation()
            self.activation = activation()
            self.in_channel = in_channel
            self.out_channel = out_channel
            self.inverse = inverse
            self.dim = dim
            self.skiplink = skiplink
            self.liftup = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0),
                self.activation
            )
            c = 2*self.fold if (not self.inverse) else 1
            if (self.skiplink):
                c = 2*c
            d = self.fold if (not self.inverse) else 1
            self.conv2d = nn.Sequential(
                nn.Conv2d(out_channel*c, hidden_channel, kernel_size=1 if (not self.inverse) else ksize, padding=0 if (not self.inverse) else padding),
                self.hidden_activation,
                nn.Conv2d(hidden_channel, hidden_channel, kernel_size=ksize, padding=padding),
                self.hidden_activation,
                nn.Conv2d(hidden_channel, out_channel*d, kernel_size=ksize, padding=padding),
                self.hidden_activation
            )
            self.project = nn.Sequential(
                nn.Conv2d(out_channel*2 if (self.dim is None or self.inverse) else out_channel, in_channel,kernel_size=ksize,padding=padding),
            )

        def forward(self, x, y=None):
            x_lift = self.liftup(x)
            if (not self.inverse):
                x = torch.fft.rfft2(x_lift)
                if(self.mode <0):
                    x = torch.concat([x[:, :, (abs(self.mode) * f):abs(self.mode) + (f * abs(self.mode)), self.mode:] for f in range(self.fold)], dim=1)
                else:
                    x = torch.concat([x[:, :, (self.mode * f):self.mode + (f * self.mode), :self.mode] for f in range(self.fold)], dim=1)
                #size = (x.shape[2],x.shape[3]) if (self.dim is None) else self.dim
                x = torch.view_as_real(x)
                x_fft = torch.concatenate([x[:,:,:,:,0],x[:,:,:,:,1]],dim=1)
                if (self.skiplink):
                    x = self.conv2d(torch.concat([x_fft,y],dim=1))
                else:
                    x = self.conv2d(x_fft)
                x = torch.concat([x[:, (self.out_channel*f):self.out_channel+(self.out_channel*f),:,:] for f in range(self.fold)], dim=2)
                x = torch.fft.irfft2(x)
                if (self.dim is None):
                    x = torch.concat([x, x_lift],dim=1)
                    x = self.project(x)
                    if (self.skiplink):
                        return x
                    else:
                        return x , x_fft
                else:
                    x = self.project(x)
                    x = self.activation(x)
                    return x , x_lift
            else:
                x = self.conv2d(x_lift)
                x = torch.fft.irfft2(x,self.dim)
                x = torch.concat([x,y],dim=1)
                x = self.project(x)
                return x

