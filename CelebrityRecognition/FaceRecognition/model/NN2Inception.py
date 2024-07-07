import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock1a(nn.Module):
    def __init__(self):
        super(InceptionBlock1a, self).__init__()
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1),
            nn.BatchNorm2d(96, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=0.00001),
            nn.ReLU()
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(192, 16, kernel_size=1),
            nn.BatchNorm2d(16, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32, eps=0.00001),
            nn.ReLU()
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(192, 32, kernel_size=1),
            nn.BatchNorm2d(32, eps=0.00001),
            nn.ReLU()
        )
        
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.BatchNorm2d(64, eps=0.00001),
            nn.ReLU()
        )
        
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        branch1x1 = self.branch1x1(x)
        
        outputs = [branch3x3, branch5x5, branch_pool, branch1x1]
        return torch.cat(outputs, 1)

class InceptionBlock1b(nn.Module):
    def __init__(self):
        super(InceptionBlock1b, self).__init__()
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(256, 96, kernel_size=1),
            nn.BatchNorm2d(96, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=0.00001),
            nn.ReLU()
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.BatchNorm2d(32, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64, eps=0.00001),
            nn.ReLU()
        )
        
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64, eps=0.00001),
            nn.ReLU()
        )
        
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64, eps=0.00001),
            nn.ReLU()
        )
        
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        branch1x1 = self.branch1x1(x)
        
        outputs = [branch3x3, branch5x5, branch_pool, branch1x1]
        return torch.cat(outputs, 1)
   
class InceptionBlock1c(nn.Module):
    def __init__(self):
        super(InceptionBlock1c, self).__init__()
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=1),
            nn.BatchNorm2d(128, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(256, eps=0.00001),
            nn.ReLU()
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(320, 32, kernel_size=1),
            nn.BatchNorm2d(32, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=(2, 2), padding=2),
            nn.BatchNorm2d(64, eps=0.00001),
            nn.ReLU()
        )

    def forward(self, X):
        branch3x3 = self.branch3x3(X)
        branch5x5 = self.branch5x5(X)

        X_pool = F.max_pool2d(X, kernel_size=3, stride=2, padding=0)
        X_pool = F.pad(X_pool, (0, 1, 0, 1))

        inception = torch.cat([branch3x3, branch5x5, X_pool], dim=1)

        return inception

class InceptionBlock2a(nn.Module):
    def __init__(self):
        super(InceptionBlock2a, self).__init__()

        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(640, 96, kernel_size=1),
            nn.BatchNorm2d(96, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(96, 192, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(192, eps=0.00001),
            nn.ReLU()
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(640, 32, kernel_size=1),
            nn.BatchNorm2d(32, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=(1, 1), padding=2),
            nn.BatchNorm2d(64, eps=0.00001),
            nn.ReLU()
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Sequential(
                    nn.Conv2d(640, 128, kernel_size=1),
                    nn.BatchNorm2d(128, eps=0.00001),
                    nn.ReLU(),
                ))
        self.branch1x1 = nn.Sequential(
                    nn.Conv2d(640, 256, kernel_size=1),
                    nn.BatchNorm2d(256, eps=0.00001),
                    nn.ReLU(),
                ) 
        
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        branch1x1 = self.branch1x1(x)
        
        outputs = [branch3x3, branch5x5, branch_pool, branch1x1]
        return torch.cat(outputs, 1)

class InceptionBlock2b(nn.Module):  #4e
    
    def __init__(self):
        super(InceptionBlock2b, self).__init__()
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=1),
            nn.BatchNorm2d(128, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(256, eps=0.00001),
            nn.ReLU()
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(640, 64, kernel_size=1),
            nn.BatchNorm2d(64, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=(2, 2), padding=2),
            nn.BatchNorm2d(128, eps=0.00001),
            nn.ReLU()
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.ZeroPad2d(padding=(0, 1, 0, 1))
        )
        
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class InceptionBlock3a(nn.Module):
    def __init__(self):
        super(InceptionBlock3a, self).__init__()

        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(1024, 192, kernel_size=1),
            nn.BatchNorm2d(192, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(192, 384, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(384, eps=0.00001),
            nn.ReLU()
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(1024, 48, kernel_size=1),
            nn.BatchNorm2d(48, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(48, 128, kernel_size=5, stride=(1, 1), padding=2),
            nn.BatchNorm2d(128, eps=0.00001),
            nn.ReLU()
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Sequential(
                    nn.Conv2d(1024, 128, kernel_size=1),
                    nn.BatchNorm2d(128, eps=0.00001),
                    nn.ReLU(),
                ))
        self.branch1x1 = nn.Sequential(
                    nn.Conv2d(1024, 384, kernel_size=1),
                    nn.BatchNorm2d(384, eps=0.00001),
                    nn.ReLU(),
                ) 
        
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        branch1x1 = self.branch1x1(x)
        
        outputs = [branch3x3, branch5x5, branch_pool, branch1x1]
        return torch.cat(outputs, 1)
    
class InceptionBlock3b(nn.Module):
    def __init__(self):
        super(InceptionBlock3b, self).__init__()
       
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(1024, 192, kernel_size=1),
            nn.BatchNorm2d(192, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(192, 384, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(384, eps=0.00001),
            nn.ReLU()
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(1024, 48, kernel_size=1),
            nn.BatchNorm2d(48, eps=0.00001),
            nn.ReLU(),
            nn.Conv2d(48, 128, kernel_size=5, stride=(1, 1), padding=2),
            nn.BatchNorm2d(128, eps=0.00001),
            nn.ReLU()
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Sequential(
                    nn.Conv2d(1024, 128, kernel_size=1),
                    nn.BatchNorm2d(128, eps=0.00001),
                    nn.ReLU(),
                ))
        self.branch1x1 = nn.Sequential(
                    nn.Conv2d(1024, 384, kernel_size=1),
                    nn.BatchNorm2d(384, eps=0.00001),
                    nn.ReLU(),
                ) 
        
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        branch1x1 = self.branch1x1(x)
        
        outputs = [branch3x3, branch5x5, branch_pool, branch1x1]
        return torch.cat(outputs, 1)

class NN2Inception2(nn.Module):
    def __init__(self):
        super(NN2Inception2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64, eps=0.00001)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(192, eps=0.00001)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_3a = InceptionBlock1a()
        self.inception_3b = InceptionBlock1b()
        self.inception_3c = InceptionBlock1c()
        
        self.inception_4a = InceptionBlock2a()
        self.inception_4b = InceptionBlock2b()
        
        self.inception_5a = InceptionBlock3a()
        self.inception_5b = InceptionBlock3b()
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(512, 256)  
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)  
        self.fc3 = nn.Linear(256, 128) 
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)  
        self.dense = nn.Linear(128, 512)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.maxpool(x)
        
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.inception_3c(x)
        
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.dense(x)
        
        x = F.normalize(x, p=2, dim=1)
        
        return x
    

class NN2Inception(nn.Module):
    def __init__(self):
        super(NN2Inception, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64, eps=0.00001)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(192, eps=0.00001)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_3a = InceptionBlock1a()
        self.inception_3b = InceptionBlock1b()
        self.inception_3c = InceptionBlock1c()
        
        self.inception_4a = InceptionBlock2a()
        self.inception_4b = InceptionBlock2b()
        
        self.inception_5a = InceptionBlock3a()
        self.inception_5b = InceptionBlock3b()
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(1024, 512)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.maxpool(x)
        
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.inception_3c(x)
        
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        
        x = F.normalize(x, p=2, dim=1)
        
        return x


model = NN2Inception().cuda()
