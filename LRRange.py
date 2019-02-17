import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


from classTinyArch import classTinyArch as architecture
# from classificationArch import classArch as architecture
from function import *
import argparse
import time
import pdb

parser = argparse.ArgumentParser(description='Yolo classification implementation Arguments')
parser.add_argument('-b', '--batchSize', default=110, type=int, metavar='N (integer)', help='mini-batchSize '
                                                                                          ',default = 40')
parser.add_argument('-lrMin', '--lrMin', default=1e-3, metavar='N (float)', type=float, help='min Learning rate 4 range test '
                                                                                      ',default = 1e-5')
parser.add_argument('-lrMax', '--lrMax', default=0.5, metavar='N (float)', type=float, help='max Learning rate 4 range test '
                                                                                      ',default = 3')
parser.add_argument('-e', '--stopEpoch', default=1, type=int, metavar='N (integer)',
                    help='epoch num to run ,default = 5')
parser.add_argument('-d', '--dir', type=str, metavar='PATH', default='/media/osmant/Data/imagenet/',
                    help='imagenetFolder')
parser.add_argument('-dev','--device', type=str, metavar='str', default='cuda', help='device to run the model ,default = cuda')
parser.add_argument('-log','--fileName',type=str,default='tiny.log',help='writeOutput2File')
parser.add_argument('-rTest','--lrRange',type=bool,default=True,help='LRRateRange')

args = parser.parse_args()
args.startEpoch = 0
args.beta = 0.9
print("batchSize {}, epoch{}".format(args.batchSize,args.stopEpoch))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trainDataset = datasets.ImageFolder(
    args.dir + 'train',
    transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

trainLoader = torch.utils.data.DataLoader(
    trainDataset, batch_size=args.batchSize, shuffle=True,
    num_workers=3, pin_memory=True, sampler=None)

ValDataSet = datasets.ImageFolder(
    args.dir + 'val',
    transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

ValLoader = torch.utils.data.DataLoader(
    ValDataSet, batch_size=args.batchSize, shuffle=True,
    num_workers=3, pin_memory=True, sampler=None)

numOfIt = int(args.stopEpoch*len(trainLoader.dataset)/args.batchSize)
# lrRange = logLinearLR(args.lrMin,args.lrMax,numOfIt+2)
lrRange = linearLR(args.lrMin,args.lrMax,numOfIt+2)


valData,valTarget= selectValData(ValLoader,args.batchSize)

net = architecture()
optimizer = optim.SGD(net.parameters(), lr=lrRange[0], momentum=0.9, weight_decay=0.0005)

net = net.to(args.device)
criterion = nn.CrossEntropyLoss().cuda()

trainLoss = []
valLoss = []
lrTrain = []
lrVall =  []

avgTrainLoss = -torch.log(torch.tensor(1/1000)).to('cuda')
avgValLoss =  -torch.log(torch.tensor(1/1000)).to('cuda')
# pdb.set_trace()
for it in range(args.startEpoch, args.stopEpoch):
    batchStart = time.time()

    for i, (inData, target) in enumerate(trainLoader):
        tempIt = int(it * len(trainLoader.dataset)/args.batchSize) + i
        optimizer.param_groups[0]['lr'] = lrRange[tempIt].cuda()
        net.train()
        target = target.cuda(non_blocking=True)
        inData = inData.cuda(non_blocking=True)

        output = net(inData)
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avgTrainLoss = movingAvg(avgTrainLoss,args.beta,loss)
        trainLoss.append(avgTrainLoss.item())
        lrTrain.append(lrRange[tempIt])
        if i % 20 == 0:
            topTrain1, topTrain5 = accuracy(output, target, topk=(1, 5))

            learRate = optimizer.param_groups[0]['lr']
            
            net.eval()

            with torch.no_grad():
                valTarget = valTarget.cuda(non_blocking=True)
                valData = valData.cuda(non_blocking=True)
                output = net(valData)
                loss = criterion(output, valTarget)
                avgValLoss = movingAvg(avgValLoss,args.beta,loss)
                valLoss.append(avgValLoss.item())
                lrVall.append(lrRange[tempIt])
                topVal1, topVal5 = accuracy(output, valTarget, topk=(1, 5))
            print(
                "epoch {}, epochProgress{}/{},lr {}, lossTrain {:.3f}, lossVal {:.3f},  accuTrain1 {:.3f} / accuVal1 {:.3f},accuTrain5 {:.3f} / accuVal5 {:.3f} , calctime{:.3f}".
                format(it, i, len(trainLoader), learRate, torch.mean(avgTrainLoss), torch.mean(avgValLoss), topTrain1.item(), topVal1.item()
                       ,topTrain5.item(), topVal5.item() , time.time() - batchStart))
            batchStart = time.time()

            plt.plot(lrTrain,trainLoss,'b')
            plt.plot(lrVall,valLoss,'r')
            plt.savefig('lrRange.png')

data2Save = {}
data2Save['trainLoss'] = trainLoss
data2Save['lrTrain'] = lrTrain
data2Save['valLoss'] = valLoss
data2Save['lrVall'] = lrVall
torch.save(data2Save,'lrRangeTest.pth')

