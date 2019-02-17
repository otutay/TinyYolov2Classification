import torch
# import pdb
import matplotlib.pyplot as plt
import numpy as np
import time

def updateTime(time2Update,timeStart):
    timePassed = time.time()-timeStart
    time2Update += timePassed 
    return time2Update

def movingAvg(oldAvg,beta,newData):
    newAvg = beta*oldAvg + (1-beta)*newData
    return newAvg

def selectValData(ValLoader,batchSize):
    numOfData = len(ValLoader.dataset)
    randLoc = torch.randperm(numOfData)[:batchSize]
    valData = torch.zeros(batchSize,3,224,224)
    valTarget = torch.zeros(batchSize,dtype=torch.long)
    for it in range(batchSize):
        valData[it] = ValLoader.dataset[randLoc[it]][0]
        valTarget[it] = ValLoader.dataset[randLoc[it]][1]

    return valData,valTarget

def createDataset(save,**kwargs):
    with torch.no_grad():
        for key,value in kwargs.items():
            mean = increaseDim(value)
            kwargs = {key:mean}
            save.createDataset(**kwargs)

def saveDataset(save,**kwargs):
    with torch.no_grad():
         for key,value in kwargs.items():
            mean = increaseDim(value)
            kwargs = {key:mean}
            save.variable2Write(**kwargs)

def saveNetworkParam(save,net):
    with torch.no_grad():
        for name,param in net.named_parameters():
            tempName = name.split('.')
            if tempName[0][-1] !='B':
                mean = calculateMeanStd(param.data)
                kwargs = {name:mean}
                save.variable2Write(**kwargs)
                mean = calculateMeanStd(param.grad.data)
                kwargs = {name+'.grad':mean}
                save.variable2Write(**kwargs)

def createNetworkParam(save,net):
    with torch.no_grad():
        for name,param in net.named_parameters():
            tempName = name.split('.')
            if tempName[0][-1] !='B':
                mean = calculateMeanStd(param.data)  
                kwargs = {name:mean}
                save.createDataset(**kwargs)
                mean = calculateMeanStd(param.grad.data)
                kwargs = {name+'.grad':mean}
                save.createDataset(**kwargs)


def calculateMeanStd(data):
    with torch.no_grad():
        if len(data.shape) == 1:
            meanData = increaseDim(data)
        elif len(data.shape)== 2:
            meanData = torch.mean(data,dim=0)
        elif len(data.shape)== 4:
            meanData = torch.mean(data,dim=1)
            meanData = increaseDim(meanData)
        return meanData


def increaseDim(data1):
    with torch.no_grad():
        return data1.unsqueeze(0)

def adjust_lr(optimizer,iter,epoch,learRate,power):
    lr = learRate*(1 - iter/epoch)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batchSize = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batchSize))
        return res


def initKernels(fileName,net):
    data = torch.load(fileName)
    oldNet = data['net']
    with torch.no_grad():
        for data1,data2 in zip(oldNet.named_parameters(),net.named_parameters()):
            tempName = data1[0].split('.')
            if tempName[0][-1] !='B':
                data2[1] .data = data1[1].data
    return net

def initWeights(net):
    with torch.no_grad():
        for data1 in net.named_parameters():
            torch.nn.init.normal_(data1[1],mean = 0,std= 0.01)
        
    return net

def cycleParam(startVals,stopVals,percents,types,totalNum):
    # import pdb;pdb.set_trace()
    param = []
    for startVal,stopVal,percent,typeR in zip(startVals,stopVals,percents,types):
        tempNum = int((totalNum*percent/100))
        if typeR == 'lin':
            param[-1:] = torch.linspace(startVal,stopVal,tempNum)
        elif typeR == 'cos':
            param[-1:] = stopVal+(startVal-stopVal)*torch.tensor(np.cos(2*np.pi*1/(tempNum)*np.linspace(0,tempNum/2,tempNum))+1)/2
    param = torch.tensor(param)
    return param.reshape((1,-1))




# def cycleParam(startVals,stopVals,percents,totalNum):
#     param = []
#     for startVal,stopVal,percent in zip(startVals,stopVals,percents):
#         tempNum = int((totalNum*percent/100))
#         param[-1:] = torch.linspace(startVal,stopVal,tempNum)
#     params = torch.tensor(param)
#     return params.reshape((1,-1))
        


def logLinearLR(startVal,stopVal,itNum):
    logStart = torch.log10(torch.tensor(startVal).type(torch.float))
    logStop = torch.log10(torch.tensor(stopVal).type(torch.float))
    tempLr = torch.logspace(logStart,logStop,itNum)
    return tempLr


def linearLR(startVal,stopVal,itNum):
    tempLr = torch.linspace(startVal,stopVal,itNum)
    return tempLr

def loadCheckpoint(fileName):
    data = torch.load(fileName)
    epoch = data['epoch']
    net = data['net']
    optim = data['optim']
    loss = data['loss']
    top1 = data['top1']
    top5 = data['top5']
    
    return epoch,net,optim,loss,top1,top5

def saveCheckPoint(fileName,epoch,net,optimizer,loss,top1,top5,isBest):
    data={}
    data['epoch'] = epoch
    data['net'] = net
    data['loss'] = loss
    data['top1'] = top1
    data['top5'] = top5
    data['optim'] = optimizer
    if isBest:
        print('data is written to disk as best')
        torch.save(data,'BestData.pth')
        torch.save(data,fileName)
    else:
        print('data is written to disk')
        torch.save(data,fileName)
    


if __name__=="__main__":
    startVals =[0.7,5,7]
    stopVals = [5  ,7,0.07]
    percents = [50,10,40]
    typeR = ['lin','lin','cos']
    param = cycleParam(startVals,stopVals,percents,typeR,500)
    print(param.shape)
    plt.figure()
    plt.plot(np.array(param).reshape((-1)))
    plt.show()

