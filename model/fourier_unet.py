#!/usr/bin/python3

import torch
import torch.nn as nn

# Reference Soruce : https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py


class FourierUNET(nn.Module):

    def __init__(self, dim, in_channel=1, project_channel=4, hidden_channel=8, pooling='max', mode=256, hiddenactivation='GELU',
                 encoder_ksize=(3,3), decoder_ksize=(3,3), encoder_padding=1, decoder_padding=1,
                 updownsample_ksize=(2,2), updownsample_padding=0, device=None):
        super().__init__()
        self.device = None
        self.mode = mode
        self.dim = dim
        self.in_channel = in_channel
        self.hiddenactivation = hiddenactivation
        self.fold = self.dim[0]//self.mode if (self.dim[0]//self.mode>0) else 1
        if (device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.poolinglayer = nn.MaxPool2d(updownsample_ksize,updownsample_padding) if (pooling == 'max') else nn.AvgPool2d(updownsample_ksize,updownsample_padding)
        self.unet_Fencoder1 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding, hiddenactivation=self.hiddenactivation, mode=self.mode, fold=self.fold).to(self.device)
        self.unet_Fencoder2 = self._Fencoder_convBlock(1, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_Fencoder3 = self._Fencoder_convBlock(1, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_Fencoder4 = self._Fencoder_convBlock(1, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_Fencoder5 = self._Fencoder_convBlock(1, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation).to(self.device)

        self.unet_Fdecoder1 = self._Fdecoder_convBlock(1, project_channel, hidden_channel, decoder_ksize, decoder_padding,
                                                       upsample_ksize=updownsample_ksize, upsample_padding=updownsample_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_Fdecoder2 = self._Fdecoder_convBlock(1, project_channel, hidden_channel, decoder_ksize, decoder_padding,
                                                       upsample_ksize=updownsample_ksize, upsample_padding=updownsample_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_Fdecoder3 = self._Fdecoder_convBlock(1, project_channel, hidden_channel, decoder_ksize, decoder_padding,
                                                       upsample_ksize=updownsample_ksize, upsample_padding=updownsample_padding, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_Fdecoder4 = self._Fdecoder_convBlock(1, project_channel, hidden_channel, decoder_ksize, decoder_padding,
                                                       upsample_ksize=updownsample_ksize, upsample_padding=updownsample_padding, concat=False, hiddenactivation=self.hiddenactivation).to(self.device)
        self.unet_Fdecoder5 = self._Fdecoder_convBlock(1, in_channel, hidden_channel, decoder_ksize, decoder_padding,
                                                      upsample_ksize=updownsample_ksize, upsample_padding=updownsample_padding,  concat=False, hiddenactivation=self.hiddenactivation, dim=self.dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        encoder, _ = self.unet_Fencoder1(x)
        #encoder = self.poolinglayer(encoder)
        encoder, encoder2_fft = self.unet_Fencoder2(encoder)
        #encoder = self.poolinglayer(encoder)
        encoder, encoder3_fft = self.unet_Fencoder3(encoder)
        #encoder = self.poolinglayer(encoder)
        encoder, encoder4_fft = self.unet_Fencoder4(encoder)
        #encoder = self.poolinglayer(encoder)
        upscale, _ = self.unet_Fencoder5(encoder)

        upscale = self.unet_Fdecoder1(upscale,encoder4_fft)
        upscale = self.unet_Fdecoder2(upscale,encoder3_fft)
        upscale = self.unet_Fdecoder3(upscale,encoder2_fft)
        upscale = self.unet_Fdecoder4(upscale)
        upscale = self.unet_Fdecoder5(upscale)

        return upscale

    class _Fencoder_convBlock(nn.Module):

        def __init__(self, in_channel, out_channel, hidden_channel, ksize=(3, 3), padding=1,
                     hiddenactivation='GELU',mode=0, fold=1):
            super().__init__()
            self.mode=mode
            self.fold = fold
            hiddenactivation = getattr(nn, hiddenactivation)
            activation = hiddenactivation()
            self.in_channel = in_channel
            self.out_channel = out_channel
            self.liftup = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0),
                activation
            )
            self.conv2d = nn.Sequential(
                nn.Conv2d(out_channel*2*self.fold, hidden_channel, kernel_size=1, padding=0),
                activation,
                nn.Conv2d(hidden_channel, hidden_channel, kernel_size=ksize, padding=padding),
                activation,
                nn.Conv2d(hidden_channel, hidden_channel, kernel_size=ksize, padding=padding),
                activation
            )
            self.project = nn.Sequential(
                nn.Conv2d(hidden_channel,in_channel,kernel_size=1,padding=0),
                activation
            )

        def forward(self, x):
            size = (x.shape[2],x.shape[3])
            x = self.liftup(x)
            x = torch.fft.rfft2(x)
            if (self.mode > 0):
                if (self.mode <= x.shape[-1]):
                    fold = self.fold if (self.fold !=0) else x.shape[2]//self.mode
                    x = torch.concat([x[:, :, (self.mode * f):self.mode + (f * self.mode), -self.mode:] for f in range(self.fold)], dim=1)
                    size = (x.shape[2],x.shape[3])
                else:
                    zeropadding = torch.zeros((x.shape[0],x.shape[1],x.shape[2],self.mode-x.shape[-1]))
                    x = torch.concat([x,zeropadding],dim=-1)
                #x = x[:,:,:self.mode,-self.mode:]
            x = torch.view_as_real(x)
            x_fft = torch.concatenate([x[:,:,:,:,0],x[:,:,:,:,1]],dim=1)
            x = self.conv2d(x_fft)
            x = torch.fft.irfft2(x,size)
            x = self.project(x)
            return x,x_fft

    class _Fdecoder_convBlock(nn.Module):

        def __init__(self, in_channel, out_channel, hidden_channel, ksize=(3,3), padding=1, concat=True,
                     upsample_ksize=(2,2), upsample_padding=0,
                     hiddenactivation='GELU', dim=None):
            super().__init__()
            hiddenactivation = getattr(nn,hiddenactivation)
            self.activation =  hiddenactivation()
            self.dim = dim
            self.concat = concat
            self.in_channel=in_channel
            self.out_channel = out_channel
            self.liftup = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0),
                self.activation
            )
            self.upscale = nn.Sequential(
                nn.ConvTranspose2d(in_channel, in_channel, kernel_size=upsample_ksize, padding=upsample_padding, stride=2),
            )
            c = 4 if (concat) else 2
            self.conv2d = nn.Sequential(
                nn.Conv2d(out_channel*c, hidden_channel, kernel_size=1, padding=0),
                self.activation,
                nn.Conv2d(hidden_channel, hidden_channel, kernel_size=ksize, padding=padding),
                self.activation,
                nn.Conv2d(hidden_channel, hidden_channel, kernel_size=ksize, padding=padding),
                self.activation,
            )
            self.project = nn.Sequential(
                nn.Conv2d(hidden_channel,in_channel,kernel_size=1,padding=0),
            )

        def forward(self, x, y=None):
            x = x if (self.concat) else self.upscale(x)
            x = self.liftup(x)
            x = torch.fft.rfft2(x)
            x = torch.view_as_real(x)
            x = torch.concat([x[:,:,:,:,0],x[:,:,:,:,1]],dim=1)
            if (self.concat):
                x = torch.concat([x,y],dim=1)
            x = self.conv2d(x)
            x = torch.fft.irfft2(x,self.dim)
            x = self.project(x)
            x = self.activation(x) if (self.dim is None) else x
            return x



