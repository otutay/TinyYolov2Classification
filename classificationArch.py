import torch.nn as nn
import torch.nn.functional as F

class classArch(nn.Module):
    def __init__(self):
        super(classArch,self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,stride=1,padding=1)
        self.conv1Batch = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv2Batch = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv3Batch = nn.BatchNorm2d(128)

        # self.conv4 = nn.Conv2d(128,64,1,padding=1)
        self.conv4 = nn.Conv2d(128,64,1)
        self.conv4Batch = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64,128,3,padding=1)
        self.conv5Batch = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128,256,3,padding=1)
        self.conv6Batch = nn.BatchNorm2d(256)

        # self.conv7 = nn.Conv2d(256,128,1,padding=1)
        self.conv7 = nn.Conv2d(256,128,1)
        self.conv7Batch = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128,256,3,padding=1)
        self.conv8Batch = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256,512,3,padding=1)
        self.conv9Batch = nn.BatchNorm2d(512)

        # self.conv10 = nn.Conv2d(512,256,1,padding=1)
        self.conv10 = nn.Conv2d(512,256,1)
        self.conv10Batch = nn.BatchNorm2d(256)

        self.conv11 = nn.Conv2d(256,512,3,padding=1)
        self.conv11Batch = nn.BatchNorm2d(512)

        # self.conv12 = nn.Conv2d(512,256,1,padding=1)
        self.conv12 = nn.Conv2d(512,256,1)
        self.conv12Batch = nn.BatchNorm2d(256)

        self.conv13 = nn.Conv2d(256,512,3,padding=1)
        self.conv13Batch = nn.BatchNorm2d(512)

        self.conv14 = nn.Conv2d(512,1024,3,padding=1)
        self.conv14Batch = nn.BatchNorm2d(1024)

        # self.conv15 = nn.Conv2d(1024,512,1,padding=1)
        self.conv15 = nn.Conv2d(1024,512,1)
        self.conv15Batch = nn.BatchNorm2d(512)

        self.conv16 = nn.Conv2d(512,1024,3,padding=1)
        self.conv16Batch = nn.BatchNorm2d(1024)

        # self.conv17 = nn.Conv2d(1024,512,1,padding=1)
        self.conv17 = nn.Conv2d(1024,512,1)
        self.conv17Batch = nn.BatchNorm2d(512)

        self.conv18 = nn.Conv2d(512,1024,3,padding=1)
        self.conv18Batch = nn.BatchNorm2d(1024)

        self.conv19 = nn.Conv2d(1024,1000,1)


    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv1Batch(x)

        x = F.max_pool2d(x,2,2)

        x = self.conv2(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv2Batch(x)

        x = F.max_pool2d(x,2,2)

        x = self.conv3(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv3Batch(x)
        x = self.conv4(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv4Batch(x)
        x = self.conv5(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv5Batch(x)

        x = F.max_pool2d(x,2,2)

        x = self.conv6(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv6Batch(x)
        x = self.conv7(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv7Batch(x)
        x = self.conv8(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv8Batch(x)

        x = F.max_pool2d(x,2,2)

        x = self.conv9(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv9Batch(x)
        x = self.conv10(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv10Batch(x)
        x = self.conv11(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv11Batch(x)
        x = self.conv12(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv12Batch(x)
        x = self.conv13(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv13Batch(x)

        x = F.max_pool2d(x,2,2)

        x = self.conv14(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv14Batch(x)
        x = self.conv15(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv15Batch(x)
        x = self.conv16(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv16Batch(x)
        x = self.conv17(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv17Batch(x)
        x = self.conv18(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv18Batch(x)

        x = self.conv19(x)
        x = F.leaky_relu(x,0.1)

        x = F.avg_pool2d(x,7,1)
        x = x.view(-1,1000)
        return x
