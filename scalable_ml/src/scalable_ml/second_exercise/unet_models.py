"""
Pytorch implementation of a Unet architecure for semantic image segmentation
"""
import torch
from torch import nn
from torchvision.models import vgg16_bn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # encoder
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # bottleneck
        self.conv_bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d( in_channels =256 , out_channels =256 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1)
            )
        
        
        #decoder
        self.up_conv1 = nn.Sequential (
            # Conv2d : here we half the number of channels
            nn.Conv2d( in_channels = 512 , out_channels = 256 , kernel_size=3 , stride =1 , padding =1) , # 2x2 kernel?
            # BatchNorm2d : num_features = out_channels
            nn.BatchNorm2d( num_features = 256) ,
            nn.ReLU(),
            # ConvTranspose2d : here we double the resolution H x W
            nn.ConvTranspose2d( in_channels =256 , out_channels =128 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1),
        )

        self.up_conv2 = nn.Sequential (
            # Conv2d : here we half the number of channels
            nn.Conv2d( in_channels = 256 , out_channels = 128 , kernel_size=3 , stride =1 , padding =1) ,
            # BatchNorm2d : num_features = out_channels
            nn.BatchNorm2d( num_features = 128) ,
            nn.ReLU(),
            # ConvTranspose2d : here we double the resolution H x W
            nn.ConvTranspose2d( in_channels =128 , out_channels =64 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1),
        )

        self.up_conv3 = nn.Sequential (
            # Conv2d : here we half the number of channels
            nn.Conv2d( in_channels = 128 , out_channels = 64, kernel_size=3 , stride =1 , padding =1) ,
            # BatchNorm2d : num_features = out_channels
            nn.BatchNorm2d( num_features = 64) ,
            nn.ReLU(),
            # ConvTranspose2d : here we double the resolution H x W
            nn.ConvTranspose2d( in_channels =64 , out_channels =32 ,kernel_size =3 , stride =2 , padding =1 , output_padding =1),
        )


        self.final_conv = nn.Sequential (
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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
        x = torch.cat([ x , block4 ] , dim =1) #skip connection
        x = self.up_conv1( x )
        
        x = torch.cat([ x , block3 ] , dim =1) #skip connection
        x = self.up_conv2( x )
        
        x = torch.cat([ x , block2 ] , dim =1) #skip connection
        x = self.up_conv3( x )
        
        x = torch.cat([ x , block1 ] , dim =1) #skip connection
        

        x = self.final_conv(x)
        
        return x



class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self, ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
