"""
An implementation of AlexNet CNN architecture.
Programmed by Saran Pannasuriyaporn (runpan4work@gmail.com)
"""
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet (Sigmoid as output activation, change if you would like)
    Expect the input size (3x227x227)
    """
    def __init__(self, in_channels = 3, num_classes = 10):
        super(AlexNet,self).__init__()
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 96, kernel_size = (11,11), stride=(4,4), padding=(0,0))
        self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride=(2,2) ) 

        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = (5,5), stride=(1,1), padding="same") #same padding
        self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride=(2,2) ) 

        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = (3,3), stride=(1,1), padding="same")
        
        self.conv4 = nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = (3,3), stride=(1,1), padding="same")

        self.conv5 = nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = (3,3), stride=(1,1), padding="same")
        self.pool5 = nn.MaxPool2d( kernel_size = (3,3), stride=(2,2) ) 

        self.fc1 = nn.Linear(9216,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,num_classes)

        self.num_classes = num_classes
        self._init_weight()

    def forward(self, x):
        x = self.relu (self.conv1(x))
        x = self.pool1(x)

        x = self.relu( self.conv2(x) )
        x = self.pool2(x)

        x = self.relu( self.conv3(x) )

        x = self.relu( self.conv4(x) )

        x = self.relu( self.conv5(x) )
        x = self.pool5(x)

        x = x.reshape(x.shape[0] , -1) #flatten
        
        x = self.relu( self.fc1(x) )
        x = self.relu( self.fc2(x) )
        x = self.sigmoid( self.fc3(x) )

        return x
    
    def _init_weight(self):
        # need kalming weight init
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def test_case():
    x = torch.randn(32, 3, 227, 227)
    model = AlexNet(in_channels = 3, num_classes = 1)
    return model(x)

if __name__ == '__main__':
    print( test_case() )
