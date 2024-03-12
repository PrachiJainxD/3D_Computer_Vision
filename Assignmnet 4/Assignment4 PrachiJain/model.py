import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        args,
        dropout_prob=0.1,
    ):
        super(Decoder, self).__init__()

        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # PReLU layers, Dropout layers and a tanh layer.
        # ***********************************************************************
        self.dropout_prob = dropout_prob 
        
        self.fc1  = nn.utils.weight_norm(nn.Linear(3, 512))     
        self.fc2  = nn.utils.weight_norm(nn.Linear(512,512))
        self.fc3  = nn.utils.weight_norm(nn.Linear(512,512))
        self.fc4  = nn.utils.weight_norm(nn.Linear(512,509))   
        self.fc5  = nn.utils.weight_norm(nn.Linear(512,512))
        self.fc6  = nn.utils.weight_norm(nn.Linear(512,512))
        self.fc7  = nn.utils.weight_norm(nn.Linear(512,512))
        self.fc8  = nn.Linear(512,1)
        
        self.prelu = nn.PReLU()       
        self.drop = nn.Dropout(dropout_prob)
        self.th = nn.Tanh()        

    # input: N x 3
    def forward(self, input):

        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        # ***********************************************************************
        
        x_copy = input
        
        x1 = self.drop(self.prelu(self.fc1(input)))
        x2 = self.drop(self.prelu(self.fc2(x1)))
        x3 = self.drop(self.prelu(self.fc3(x2)))
        x4 = self.drop(self.prelu(self.fc4(x3)))
        
        x4 = torch.cat((x4, x_copy), dim = 1)
        
        x5 = self.drop(self.prelu(self.fc5(x4)))
        x6 = self.drop(self.prelu(self.fc6(x5)))
        x7 = self.drop(self.prelu(self.fc7(x6)))
        
        x8 = self.fc8(x7)
        out = self.th(x8)
        return out
