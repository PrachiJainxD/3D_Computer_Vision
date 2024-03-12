import torch.nn as nn

class MaskedCNN(nn.Conv2d):
    """
    Masked convolution as explained in the PixelCNN variant of
    Van den Oord et al, “Pixel Recurrent Neural Networks”, NeurIPS 2016
    It inherits from Conv2D (uses the same parameters, plus the option to select a mask including
    the center pixel or not, as described in class and in the Fig. 2 of the above paper)
    """

    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A':
            self.mask[:, :, height//2, width//2:] = 0
            self.mask[:, :, height//2+1:, :] = 0
        else:
            self.mask[:, :, height//2, width//2+1:] = 0
            self.mask[:, :, height//2+1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedCNN, self).forward(x)


class PixelCNN(nn.Module):
    """
    A PixelCNN variant you have to implement according to the instructions
    """

    def __init__(self):
        super(PixelCNN, self).__init__()

        # WRITE CODE HERE TO IMPLEMENT THE MODEL STRUCTURE

        #self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        #self.sigmoid = nn.Sigmoid()
        
        # No need for Bias since there is a batchnorm layer following the CNN which removes any bias and makes it redundant
        self.layer1 = MaskedCNN(mask_type = 'A', in_channels = 1, out_channels = 16, kernel_size = (3, 3), stride = 1, padding = 'same', dilation = 3, bias = False, padding_mode = 'reflect')
        self.layer2 = nn.BatchNorm2d(16)
        self.layer3 = nn.LeakyReLU(negative_slope = 0.001)
        
        # No need for Bias since there is a batchnorm layer following the CNN which removes any bias and makes it redundant
        self.layer4 = MaskedCNN(mask_type = 'B',in_channels = 16, out_channels = 16, kernel_size= (3, 3), stride = 1, padding = 'same', dilation = 3, bias=False, padding_mode = 'reflect')
        self.layer5 = nn.BatchNorm2d(16)
        self.layer6 = nn.LeakyReLU(negative_slope = 0.001)
        
        # No need for Bias since there is a batchnorm layer following the CNN which removes any bias and makes it redundant
        self.layer7 = MaskedCNN(mask_type = 'B',in_channels = 16, out_channels = 16, kernel_size= (3, 3), stride = 1, padding = 'same', dilation = 3, bias = False, padding_mode = 'reflect')
        self.layer8 = nn.BatchNorm2d(16)
        self.layer9 = nn.LeakyReLU(negative_slope=0.001)
        
        # Use Bias since there is an activation (non-linearity) layer and no following batchnorm layer immediately after in CNN
        self.layer10 = nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = (1, 1), stride = 1, padding = 'same', bias = True)
        self.layer11 = nn.Sigmoid()        

    def forward(self, x):

        # WRITE CODE HERE TO IMPLEMENT THE FORWARD PASS
        # Mask Center and corners or just center
        A1 = self.layer1(x)
        A2 = self.layer2(A1)
        A3 = self.layer3(A2)

        # Mask Corners
        B11 = self.layer4(A3)
        B12 = self.layer5(B11)
        B13 = self.layer6(B12)

        # Mask Corners
        B21 = self.layer7(B13)
        B22 = self.layer8(B21)
        B23 = self.layer9(B22)

        convOut = self.layer10(B23)
        out = self.layer11(convOut)
        return out
