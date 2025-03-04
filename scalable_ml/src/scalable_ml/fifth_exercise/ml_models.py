"""
Pytorch implementation of a fully connected neural network and a convolutional neural network
"""
from torch import nn
import torch

class FullyConnectedNeuralNetwork(nn.Module):

    def __init__(self, ninput, noutput, nr_of_neurons=1000, nr_of_layers=1, dropout_rate=1e-5):
        super().__init__()

        self.nr_of_layers = nr_of_layers

        self.input_layer = nn.Sequential(
            nn.Linear(ninput, nr_of_neurons),
            nn.ReLU())

        self.layer_list = []
        for i in range(nr_of_layers):
            self.layer_list.append(nn.Linear(nr_of_neurons, nr_of_neurons))
            if dropout_rate > 0:
                self.layer_list.append(nn.Dropout(dropout_rate))
            self.layer_list.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*self.layer_list)

        self.output_layer = nn.Sequential(
            nn.Linear(nr_of_neurons, noutput)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x

class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self, img_dim, out_features, in_channels=1, in_conv_channels=32, n_conv_layers=1, dropout_rate=1e-5, batchnorm=True, kernel_size=3, padding=1, nr_of_neurons_fcnn=500, nr_of_layers_fcnn=3 ):
        super().__init__()

        #self.layer_list = []
        in_channels_aux=in_channels
        self.layer_list = nn.ModuleList()
        for i in range(n_conv_layers):
            self.layer_list.append(nn.Conv2d(in_channels=in_channels_aux, out_channels=in_conv_channels, kernel_size=kernel_size, padding=padding))
            if dropout_rate > 0:
                self.layer_list.append(nn.Dropout(dropout_rate))
            if batchnorm: self.layer_list.append(nn.BatchNorm2d(in_conv_channels))
            self.layer_list.append(nn.ReLU())
            self.layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels_aux=in_conv_channels
            in_conv_channels*=2

        self.conv_layers = nn.Sequential(*self.layer_list)
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_dim, img_dim)
            dummy_output = self.conv_layers(dummy_input)
            ninput = dummy_output.numel() // dummy_output.size(0) # divide by the batch size
        self.fcnn=FullyConnectedNeuralNetwork(ninput, out_features, nr_of_neurons=nr_of_neurons_fcnn, nr_of_layers=nr_of_layers_fcnn, dropout_rate=dropout_rate)

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1) #flattening the 
        out = self.fcnn(out)
        return out


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dropout_rate, batchnorm):
        super().__init__()
        self.layer_list = nn.ModuleList()
        
        self.layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer_list.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding))
        if dropout_rate > 0:
            self.layer_list.append(nn.Dropout(dropout_rate))
        if batchnorm: self.layer_list.append(nn.BatchNorm2d(out_channels))
        self.layer_list.append(nn.ReLU())
        
        self.unet_down = nn.Sequential(*self.layer_list)
                               
    def forward(self, x):
        out = self.unet_down(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, padding, dropout_rate, batchnorm):
        super().__init__()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=padding))
        if dropout_rate > 0:
            self.layer_list.append(nn.Dropout(dropout_rate))
        if batchnorm: self.layer_list.append(nn.BatchNorm2d(mid_channels))
        self.layer_list.append(nn.ReLU())
        self.layer_list.append(nn.ConvTranspose2d( in_channels=mid_channels , out_channels =out_channels, kernel_size=kernel_size , stride =2 , padding =1 , output_padding =1))

        self.unet_up = nn.Sequential(*self.layer_list)
                               
    def forward(self, x):
        out = self.unet_up(x)
        return out
    
class UNet(nn.Module):
    def __init__(self,  in_channels, out_channels, n_layers=3, in_conv_channels=32, kernel_size=3, padding=1, dropout_rate=1e-5, batchnorm=True  ):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_conv_channels, kernel_size=kernel_size, padding=1, stride=1),
            nn.BatchNorm2d(in_conv_channels),)

        in_channels_aux=in_conv_channels
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(n_layers):
            self.encoder.append( UNetBlock(in_channels_aux, 2*in_channels_aux, kernel_size, padding, dropout_rate, batchnorm))
            self.decoder.insert(0,UNetUpBlock(4*in_channels_aux, 2*in_channels_aux, in_channels_aux, kernel_size, padding, dropout_rate, batchnorm))
            in_channels_aux*=2

        self.conv_bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=in_channels_aux, out_channels=in_channels_aux, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(in_channels_aux),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=in_channels_aux, out_channels=in_channels_aux, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
            )
        
        self.final_conv = nn.Sequential(
            #nn.Conv2d(in_channels=in_conv_channels//2, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.Conv2d(in_channels=2*in_conv_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),)

    def forward(self, x):
        # encoder
        x=self.initial_conv(x)
        blocks = [x]
        for down in self.encoder:
            x = down(x)
            blocks.append(x)

        x = self.conv_bottleneck(x)
            
        for i, up in enumerate(self.decoder):
            x = torch.cat([ x , blocks[-1-i] ] , dim =1) #skip connections
            x = up( x )

        x = torch.cat([ x , blocks[0] ] , dim =1)
        return self.final_conv(x)



class ConvAutoEncoder(nn.Module):
    def __init__(self, n_channels, n_conv_layers=2, batch_norm=True):
        super().__init__()

        initial_channels=32
        activation=nn.ReLU()
        #activation=nn.ELU()
        
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



