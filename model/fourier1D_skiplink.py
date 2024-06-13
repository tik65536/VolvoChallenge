#!/usr/bin/python3

import torch
import torch.nn as nn

# Reference Soruce : https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py


class Fourier_skiplink(nn.Module):

    def __init__(self, dim, in_channel=1, project_channel=4, hidden_channel=8, pooling='max', mode=256,
                 hiddenactivation='GELU', activation='GELU', encoder_ksize=(3, 3),
                 encoder_padding=1, fftActivate=True, outputActivate=True, device=None, maxFold=3, concatlowhighFreq=False,
                 fftFilter_v=0, fftFilter_h=0, flipV = False):
        super().__init__()
        self.device = None
        self.mode = mode
        self.dim = dim
        self.in_channel = in_channel
        self.project_channel = project_channel
        self.hiddenactivation = hiddenactivation
        self.fold = 0
        if (device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.unet_Fencoder1 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, dim=self.dim, outputActivate=fftActivate, concatlowhighFreq=concatlowhighFreq,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)
        self.unet_Fencoder2 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, outputActivate=fftActivate, concatlowhighFreq=concatlowhighFreq,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)
        self.unet_Fencoder3 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, outputActivate=fftActivate, concatlowhighFreq=concatlowhighFreq,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)
        self.unet_Fencoder4 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, outputActivate=fftActivate, concatlowhighFreq=concatlowhighFreq,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)
        self.unet_Fencoder5 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, outputActivate=fftActivate, concatlowhighFreq=concatlowhighFreq,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)
        self.unet_Fencoder6 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, outputActivate=fftActivate, concatlowhighFreq=concatlowhighFreq,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)
        self.unet_Fencoder7 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, skiplink=True, outputActivate=fftActivate, concatlowhighFreq=concatlowhighFreq,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)
        self.unet_Fencoder8 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, skiplink=True, outputActivate=fftActivate, concatlowhighFreq=concatlowhighFreq,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)
        self.unet_Fencoder9 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, skiplink=True, outputActivate=fftActivate, concatlowhighFreq=concatlowhighFreq,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)
        self.unet_Fencoder10 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, skiplink=True, outputActivate=fftActivate, concatlowhighFreq=concatlowhighFreq,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)
        self.unet_Fencoder11 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, activation=activation, mode=self.mode,
                                                       fold=self.fold, skiplink=True, outputActivate=fftActivate, concatlowhighFreq=concatlowhighFreq,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)
        self.unet_Fencoder12 = self._Fencoder_convBlock(in_channel, project_channel, hidden_channel, encoder_ksize, encoder_padding,
                                                       hiddenactivation=self.hiddenactivation, mode=self.mode, fold=self.fold,
                                                       inverse=True, dim=self.dim, outputActivate=outputActivate,
                                                       fftFilter_v=fftFilter_v, fftFilter_h=fftFilter_h, flipV=flipV).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        encoder, x1 = self.unet_Fencoder1(x)
        encoder, x2 = self.unet_Fencoder2(encoder)
        encoder, x3 = self.unet_Fencoder3(encoder)
        encoder, x4 = self.unet_Fencoder4(encoder)
        encoder, x5 = self.unet_Fencoder5(encoder)
        encoder, x6 = self.unet_Fencoder6(encoder)

        encoder, _ = self.unet_Fencoder7(encoder, x6)
        encoder, _ = self.unet_Fencoder8(encoder, x5)
        encoder, _ = self.unet_Fencoder9(encoder, x4)
        encoder, _ = self.unet_Fencoder10(encoder, x3)
        encoder, _ = self.unet_Fencoder11(encoder, x2)
        encoder = self.unet_Fencoder12(encoder, x1)

        return encoder

    class _Fencoder_convBlock(nn.Module):

        def __init__(self, in_channel, out_channel, hidden_channel, ksize=3, padding=1 ,
                     hiddenactivation='GELU', activation='GELU', mode=0, fold=1, inverse=False,
                     dim=None, skiplink=False, outputActivate=True, concatlowhighFreq=False, fftFilter_v=0, fftFilter_h=0, flipV=False):
            super().__init__()
            self.concatlowhighFreq = concatlowhighFreq
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
            self.outputActivate = outputActivate
            self.fftFilter_v = fftFilter_v
            self.fftFilter_h = fftFilter_h
            self.flipV = flipV
            self.liftup = [
                            nn.Conv1d(in_channel,out_channel,kernel_size=1,padding=0),
                            self.activation
                        ]
            c = 4 if (self.skiplink) else 2
            self.conv1d = nn.Sequential(
                nn.Conv1d(self.mode*c if (not self.inverse) else self.out_channel, hidden_channel, kernel_size=1 if (not self.inverse) else ksize, padding=0 if (not self.inverse) else padding),
                self.hidden_activation,
                nn.Conv1d(hidden_channel, hidden_channel, kernel_size=ksize, padding=padding),
                self.hidden_activation,
                nn.Conv1d(hidden_channel, self.mode if (not self.inverse) else self.out_channel, kernel_size=ksize, padding=padding),
                self.hidden_activation
            )
            self.project = [
                                nn.Conv1d(out_channel*2 if (self.dim is None or self.inverse) else out_channel, in_channel,kernel_size=ksize,padding=padding),
                                self.activation
                            ]
            if (not self.outputActivate):
                print('remove activation')
                self.project = self.project[:-1]
                self.liftup = self.liftup[:-1]
            self.project = nn.Sequential(*self.project)
            self.liftup = nn.Sequential(*self.liftup)

        def forward(self, x, y=None):
            x_lift = self.liftup(x)
            if (not self.inverse):
                x = torch.fft.rfft(x_lift)
                if (self.mode <0):
                    x = x[:, :, self.mode:]
                elif (self.mode>0):
                    x = x[:, :, :self.mode]
                x = torch.view_as_real(x)
                x_fft = torch.concatenate([x[:,:,:,0],x[:,:,:,1]],dim=2)
                x_fft = x_fft.transpose(1, 2).contiguous()
                if (self.skiplink):
                    y = y.transpose(1,2).contiguous()
                    x = self.conv1d(torch.concat([x_fft,y],dim=1))
                else:
                    x = self.conv1d(x_fft)
                #x = torch.concat([x[:, (self.out_channel*f):self.out_channel+(self.out_channel*f),:,:] for f in range(self.fold)], dim=2)
                x = x.transpose(1, 2).contiguous()
                x = torch.fft.irfft(x)
                x = torch.concat([x, x_lift],dim=1) if (self.dim is None) else x
                x = self.project(x)
                r = None
                if (self.dim is None):
                    if (not self.skiplink):
                        r = x_fft.transpose(1,2).contiguous()
                else:
                        r = x_lift
                return x , r

            else:
                x = self.conv1d(x_lift)
                x = torch.fft.irfft(x,self.dim)
                x = torch.concat([x,y],dim=1)
                x = self.project(x)
                if (self.outputActivate):
                    x = self.activation(x)
                return x

