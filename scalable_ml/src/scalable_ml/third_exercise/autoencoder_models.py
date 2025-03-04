"""
Pytorch implementation of a convolutional autoencoder
"""
import torch
from torch import nn
from torchvision.models import vgg16_bn
import torch.nn.functional as F
from collections import OrderedDict


class ConvAutoEncoder2(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
  
        # encoder
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=3, padding=1, stride=1), 
            #nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # bottleneck
        self.conv_bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d( in_channels =256 , out_channels =256 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1)
            )
        
        
        #decoder
        self.up_conv1 = nn.Sequential (
            # Conv2d : here we half the number of channels
            nn.Conv2d( in_channels = 256 , out_channels = 128 , kernel_size=3 , stride =1 , padding =1) , # 2x2 kernel?
            # BatchNorm2d : num_features = out_channels
            #nn.BatchNorm2d( num_features = 256) ,
            nn.ReLU(),
            # ConvTranspose2d : here we double the resolution H x W
            nn.ConvTranspose2d( in_channels =128 , out_channels =128 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1),
        )

        self.up_conv2 = nn.Sequential (
            # Conv2d : here we half the number of channels
            nn.Conv2d( in_channels = 128 , out_channels = 64 , kernel_size=3 , stride =1 , padding =1) ,
            # BatchNorm2d : num_features = out_channels
            # nn.BatchNorm2d( num_features = 128) ,
            nn.ReLU(),
            # ConvTranspose2d : here we double the resolution H x W
            nn.ConvTranspose2d( in_channels = 64 , out_channels =64 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1),
        )

        self.up_conv3 = nn.Sequential (
            # Conv2d : here we half the number of channels
            nn.Conv2d( in_channels = 64 , out_channels = 32, kernel_size=3 , stride =1 , padding =1) ,
            # BatchNorm2d : num_features = out_channels
            #nn.BatchNorm2d( num_features = 64) ,
            nn.ReLU(),
            # ConvTranspose2d : here we double the resolution H x W
            nn.ConvTranspose2d( in_channels =32 , out_channels =32 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1),
        )


        self.final_conv = nn.Sequential (
            nn.Conv2d(in_channels=32, out_channels=n_channels, kernel_size=3, padding=1),
            #nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # encoder
        block1 = self.down_conv1( x )
        block2 = self.down_conv2( nn.MaxPool2d(kernel_size=2, stride=2)(block1) )
        block3 = self.down_conv3( nn.MaxPool2d(kernel_size=2, stride=2)(block2) )
        block4 = self.down_conv4( nn.MaxPool2d(kernel_size=2, stride=2)(block3) )
        
        # bottleneck
        x = self.conv_bottleneck( block4 )
        
        # decoder part : start on level 4 and go up again
        x = self.up_conv1( x )
        x = self.up_conv2( x )
        x = self.up_conv3( x )
        x = self.final_conv(x)
        
        return x


    
class ConvAutoEncoder3(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=3, padding=1, stride=1), 
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # bottleneck
        self.conv_bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d( in_channels =256 , out_channels =256 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1)
            )
        
        
        #decoder
        self.decoder = nn.Sequential (
            nn.Conv2d( in_channels = 256 , out_channels = 128 , kernel_size=3 , stride =1 , padding =1) , # 2x2 kernel?
            #nn.BatchNorm2d( num_features = 256) ,
            nn.ReLU(),
            nn.ConvTranspose2d( in_channels =128 , out_channels =128 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1),
            
            nn.Conv2d( in_channels = 128 , out_channels = 64 , kernel_size=3 , stride =1 , padding =1) ,
            # nn.BatchNorm2d( num_features = 128) ,
            nn.ReLU(),
            nn.ConvTranspose2d( in_channels = 64 , out_channels =64 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1),

            nn.Conv2d( in_channels = 64 , out_channels = 32, kernel_size=3 , stride =1 , padding =1),
            #nn.BatchNorm2d( num_features = 64) ,
            nn.ReLU(),
            nn.ConvTranspose2d( in_channels =32 , out_channels =32 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1),

            nn.Conv2d(in_channels=32, out_channels=n_channels, kernel_size=3, padding=1),
            #nn.ReLU(),
            nn.Sigmoid(),
        )


    def forward(self, x):
        # encoder
        block1 = self.encoder(x)
        block2 = self.conv_bottleneck(block1)
        x = self.decoder(block2)
        
        return x


class ConvAutoEncoder4(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        
        # encoder
        encoderdict= OrderedDict()
        encoderdict["cov1"] = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=3, padding=1, stride=1)
        encoderdict["relu1"] = nn.ReLU()
        encoderdict["maxpool1"] = nn.MaxPool2d(kernel_size=2, stride=2)
        encoderdict["cov2"] = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        encoderdict["relu2"] = nn.ReLU()
        encoderdict["maxpool2"] = nn.MaxPool2d(kernel_size=2, stride=2)
        encoderdict["cov3"] = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        encoderdict["relu3"] = nn.ReLU()
        encoderdict["maxpool3"] = nn.MaxPool2d(kernel_size=2, stride=2)
        encoderdict["cov4"] = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        encoderdict["relu4"] = nn.ReLU()

        #bottleneck
        bottledict = OrderedDict()
        bottledict["maxpool4"] = nn.MaxPool2d(kernel_size=2, stride=2)
        bottledict["cov5"] = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        bottledict["relu5"] = nn.ReLU()
        bottledict["covtrans"] = nn.ConvTranspose2d( in_channels =256 , out_channels =256 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1)
        bottledict["relu6"] = nn.ReLU()
        
        # decoder
        decoderdict= OrderedDict()
        decoderdict["cov7"] = nn.Conv2d( in_channels = 256 , out_channels = 128 , kernel_size=3 , stride =1 , padding =1)
        decoderdict["relu7"] = nn.ReLU()
        decoderdict["convtrans1"] = nn.ConvTranspose2d( in_channels =128 , out_channels =128 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1)

        decoderdict["cov8"] = nn.Conv2d( in_channels = 128 , out_channels = 64 , kernel_size=3 , stride =1 , padding =1) 
        decoderdict["relu8"] = nn.ReLU()
        decoderdict["convtrans2"] = nn.ConvTranspose2d( in_channels = 64 , out_channels =64 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1)

        decoderdict["cov9"] = nn.Conv2d( in_channels = 64 , out_channels = 32, kernel_size=3 , stride =1 , padding =1)
        decoderdict["relu9"] = nn.ReLU()
        decoderdict["convtrans3"] = nn.ConvTranspose2d( in_channels =32 , out_channels =32 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1)

        decoderdict["cov10"] = nn.Conv2d(in_channels=32, out_channels=n_channels, kernel_size=3, padding=1)
        decoderdict["sigmoid"] = nn.Sigmoid()
        
        # encoder
        self.encoder = nn.Sequential(encoderdict)
        
        # bottleneck
        self.conv_bottleneck = nn.Sequential(bottledict)

        # decoder
        self.decoder = nn.Sequential(decoderdict)


    def forward(self, x):
        # encoder
        block1 = self.encoder(x)
        block2 = self.conv_bottleneck(block1)
        x = self.decoder(block2)
        
        return x


class ConvAutoEncoder(nn.Module):
    def __init__(self, n_channels, n_conv_layers=2, batch_norm=True):
        super().__init__()

        initial_channels=32
        #activation=nn.ReLU()
        activation=nn.ELU()
        
        # encoder
        encoderdict= OrderedDict()

        encoderdict["cov0"] = nn.Conv2d(in_channels=n_channels, out_channels=initial_channels, kernel_size=3, padding=1, stride=1)
        if batch_norm: encoderdict["batch_norm0"] = nn.BatchNorm2d(initial_channels)
        encoderdict["relu0"] = activation
        encoderdict["maxpool0"] = nn.MaxPool2d(kernel_size=2, stride=2)
        for i in range(1, n_conv_layers):
            encoderdict["cov%i"%(i)] = nn.Conv2d(in_channels=initial_channels, out_channels=2*initial_channels, kernel_size=3, padding=1, stride=1)
            if batch_norm: encoderdict["batch_norm%i"%(i)] = nn.BatchNorm2d(2*initial_channels)
            encoderdict["relu%i"%(i)] = activation
            encoderdict["maxpool%i"%(i)] = nn.MaxPool2d(kernel_size=2, stride=2)
            initial_channels *= 2

        #bottleneck
        bottledict = OrderedDict()
        bottledict["cov_neck"] = nn.Conv2d(in_channels=initial_channels, out_channels=initial_channels, kernel_size=3, padding=1, stride=1)
        if batch_norm: bottledict["batch_norm"] = nn.BatchNorm2d(initial_channels)
        bottledict["relu_neck1"] = activation
        #bottledict["linear1"] = nn.Linear(in_features=initial_channels, out_features=initial_channels)
        #bottledict["relu_neck2"] = activation
        #bottledict["linear2"] = nn.Linear(in_features=initial_channels, out_features=initial_channels)
        #bottledict["relu_neck3"] = activation
        

        
        # decoder
        decoderdict= OrderedDict()
        decoderdict["cov_trans"] = nn.ConvTranspose2d(in_channels=initial_channels, out_channels=initial_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        for i in range(1, n_conv_layers):
            decoderdict["cov%i"%(i)] = nn.Conv2d(in_channels=initial_channels, out_channels= initial_channels , kernel_size=3 , stride =1 , padding =1)
            if batch_norm: decoderdict["batch_norm%i"%(i)] = nn.BatchNorm2d(initial_channels)
            decoderdict["relu%i"%(i)] = nn.ReLU()
            decoderdict["convtrans%i"%(i)] = nn.ConvTranspose2d(in_channels=initial_channels, out_channels=initial_channels//2, kernel_size=3, stride=2, padding=1, output_padding =1)
            initial_channels //= 2
        decoderdict["cov%i"%(n_conv_layers)] = nn.Conv2d(in_channels=initial_channels, out_channels=n_channels, kernel_size=3, padding=1)
        decoderdict["sigmoid"] = nn.Sigmoid()
        
        # encoder
        self.encoder = nn.Sequential(encoderdict)
        
        # bottleneck
        self.bottleneck = nn.Sequential(bottledict)

        # decoder
        self.decoder = nn.Sequential(decoderdict)


    def forward(self, x):
        # encoder
        block1 = self.encoder(x)
        block2 = self.bottleneck(block1)
        x = self.decoder(block2)
        
        return x
