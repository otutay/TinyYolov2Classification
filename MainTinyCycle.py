import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,ConcatDataset
from classTinyArch import classTinyArch as architecture
from function import *
from Save import Save 
import argparse
import time
# import pdb

parser = argparse.ArgumentParser(description='Yolo classification implementation Arguments')
parser.add_argument('-b', '--batchSize', default=210, type=int, metavar='N (integer)', help='mini-batchSize '
                                                                                          ',default = 210')
parser.add_argument('-e', '--stopEpoch', default=40, type=int, metavar='N (integer)',help='epoch num to run ,default = 40')
parser.add_argument('-lr', '--lr', default=0.012, type=float, metavar='N (float)',help='default learning rate, default = 0.01')
parser.add_argument('-d', '--dir', type=str, metavar='PATH', default='/imagenet/',help='imagenetFolder')
parser.add_argument('-dev','--device', type=str, metavar='str', default='cuda', help='device to run the model ,default = cuda')
parser.add_argument('-log','--fileName',type=str,default='tiny.log',help='writeOutput2File')
parser.add_argument('-res','--resume', type=bool, metavar='bool', default=False, help='resume From checkpoint ,default = False')
parser.add_argument('-load','--data2Load',type=str,default='model.pth',help='writeOutput2File')
parser.add_argument('-testEvery','--testEvery',type = int,default = 1300, help='test model in every default iteration,default = 1300')
parser.add_argument('-printEvery','--printEvery',type = int,default = 100, help='print train output in every default iteration,default = 200')
parser.add_argument("-bigImage","--bigImage",type = bool,default = False, help='image size is (448,448),default = False')


args = parser.parse_args()
args.startEpoch = 0
args.beta = 0.9
args.saveName ='yoloParameters'

save = Save(args.saveName)

print("Cycle YOLO Classification batchSize {}, epoch{}".format(args.batchSize,args.stopEpoch))
write2File = "Cycle YOLO Classification batchSize {}, epoch{}".format(args.batchSize,args.stopEpoch)


if args.fileName :
    if args.resume:
        fid = open(args.fileName,'a')
    else:
        fid = open(args.fileName,'w')
        save.createFile()
    fid.write(write2File)

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

trainDataset1 = datasets.ImageFolder(
    args.dir + 'train',
    transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation((-10,10)),
        transforms.ColorJitter(brightness = 1,contrast = 1,saturation = 1,hue = 0.3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # normalize,
    ]))

trainDataset2 = datasets.ImageFolder(
    args.dir + 'train',
    transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        # normalize,
    ]))

trainLoader = DataLoader(ConcatDataset([trainDataset1,trainDataset2])
    ,batch_size=args.batchSize, shuffle=True,
    num_workers=8, pin_memory=True, sampler=None)


ValDataSet = datasets.ImageFolder(
    args.dir + 'val',
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # normalize,
    ]))

ValLoader = DataLoader(ValDataSet, batch_size=args.batchSize, shuffle=True,
    num_workers=8, pin_memory=True, sampler=None)


if args.resume:
    args.startEpoch,net,optimizer,bestValLoss,bestTop1,bestTop5 = loadCheckpoint(args.data2Load)
    # optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9, weight_decay=0.0005)
else:
    net = architecture(args.bigImage)
    # net = initWeights(net)
    net = torch.nn.DataParallel(net).cuda()
    # optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    bestValLoss = 1000
    bestTop1 = 0
    bestTop5 = 0


numOfIt = int((args.stopEpoch)*len(trainLoader.dataset)/args.batchSize)
lrCycle = cycleParam([args.lr,args.lr,args.lr*2],[args.lr,args.lr*2,args.lr/10],[50,10,40],['lin','lin','cos'],numOfIt+args.stopEpoch)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(lrCycle.reshape((-1)).numpy())
# plt.show()

# MCycle =  cycleParam([0.9,0.8],[0.9,0.8],[50,50],numOfIt+args.stopEpoch)


if args.device == 'cuda':
    net = net.to(args.device)
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()


for it in range(args.startEpoch,args.stopEpoch):
    timeStart = time.time()
    for i, (inData, target) in enumerate(trainLoader):
        net.train()
        tempIt = int(it*len(trainLoader.dataset)/args.batchSize) + i
        optimizer.param_groups[0]['lr'] = lrCycle[0,tempIt].cuda()
        # optimizer.param_groups[0]['momentum'] = MCycle[0,tempIt].cuda()
        target = target.cuda(non_blocking=True)
        inData = inData.cuda(non_blocking=True)
        output = net(inData)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i == 0 and it == 0:
            top1, top5 = accuracy(output, target, topk=(1, 5))
            # createNetworkParam(save,net)
            createDataset(save,lr= torch.tensor(optimizer.param_groups[0]['lr']),loss=loss.clone().detach(),top1 = top1.clone().detach(),top5 = top5.clone().detach())
            
        elif i % 100 == 0:
            top1, top5 = accuracy(output, target, topk=(1, 5))
            # saveNetworkParam(save,net)
            saveDataset(save, lr = torch.tensor(optimizer.param_groups[0]['lr']),loss=loss.clone().detach(),top1 = top1.clone().detach(),top5 = top5.clone().detach())
        

        if i % args.printEvery == 0:
            top1, top5 = accuracy(output, target, topk=(1, 5))
            learRate = optimizer.param_groups[0]['lr']
            momentum = 0.0
            print(
                "epoch {}, epochProgress{}/{},lr {:.7f} / momentum {:.7f}, loss{:.5f} / bestLoss {:.3f}, accu1 {:.3f}/ bestTop1 {:.3f}, accu5 {:.3f}/ bestTop5 {:.3f}, allTime {:.3f}".
                format(it, i, len(trainLoader), learRate, momentum, loss.item(), bestValLoss, top1.item(), bestTop1, top5.item(),
                       bestTop5, time.time()-timeStart))
            if args.fileName:
                # write2File = "epoch {}, epochProgress{}/{},lr {:.7f}, loss{:.3f} / bestLoss {:.3f}, accu1 {:.3f}/ bestTop1 {:.3f}, accu5 {:.3f}/ bestTop5 {:.3f}, calctime{:.3f} \n".format(it, i, len(trainLoader), learRate, loss.item(), bestValLoss, top1.item(), bestTop1, top5.item(), bestTop5, time.time() - batchStart)
                write2File = "epoch {}, epochProgress{}/{},lr {:.7f} / momentum {:.7f}, loss{:.5f} / bestLoss {:.3f}, accu1 {:.3f}/ bestTop1 {:.3f}, accu5 {:.3f}/ bestTop5 {:.3f}, allTime {:.3f} \n".format(it, i, len(trainLoader), learRate, momentum, loss.item(), bestValLoss, top1.item(), bestTop1, top5.item(),bestTop5, time.time()-timeStart)
                fid.write(write2File)
                fid.flush()
            timeStart = time.time()

        if i % args.testEvery == 0:
            lossVal = 0
            top1Val = 0
            top5Val = 0
            net.eval()
            for ik,(inData, target) in enumerate(ValLoader):
                with torch.no_grad():
                    target = target.cuda(non_blocking=True)
                    inData = inData.cuda(non_blocking=True)
                    output = net(inData)

                    loss = criterion(output, target)
                    lossVal += loss.item()
                    top1, top5 = accuracy(output, target, topk=(1, 5))

                    top1Val += top1.item()
                    top5Val += top5.item()

            args.testNum = ik+1
            if args.fileName:
                write2File = "valLossMean{:.4f}, accuracy1Mean{:.4f},accuracy5Mean{:.4f} \n".format(lossVal / args.testNum, top1Val / args.testNum,
                                                                            top5Val / args.testNum)
                fid.write(write2File)
                fid.flush()
                print("valLossMean{:.4f}, accuracy1Mean{:.4f},accuracy5Mean{:.4f} ".format(lossVal / args.testNum, top1Val / args.testNum,
                                                                            top5Val / args.testNum))
            if i == 0 and it == 0:
                createDataset(save,ValLoss=torch.tensor(lossVal / args.testNum),valTop1 =torch.tensor(top1Val / args.testNum),valTop5 =torch.tensor(top5Val / args.testNum))
            else:
                saveDataset(save,ValLoss=torch.tensor(lossVal / args.testNum),valTop1 =torch.tensor(top1Val / args.testNum),valTop5 =torch.tensor(top5Val / args.testNum))

            if bestTop5 < top5Val/args.testNum:
                bestValLoss = lossVal/args.testNum
                bestTop1 = top1Val/args.testNum
                bestTop5 = top5Val/args.testNum
                saveCheckPoint('tinyCheckPoint.pth',it,net,optimizer,bestValLoss,bestTop1,bestTop5,True)
            else:
                saveCheckPoint('tinyCheckPoint.pth',it,net,optimizer,bestValLoss,bestTop1,bestTop5,False)

            timeStart = time.time()

