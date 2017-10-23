
import numpy as np

def  infoEntropy(Pci):                                  #计算信息熵，Pci是P(ci)，ci出现的概率，这里是个数组，i有多少数组有多少元素
    entropy = 0.0
    for P in Pci:
        entropy += -P *np.log2(P)
    return entropy

def infoGain(dataMat,j,m,n):                              #计算给定特征的信息增益
    gain=0.0
    lables = [data[-1] for data in dataMat]
    #print("infoGain")
    #print(dataMat)
    tem=sum(lables)/m
    #print(lables,m,tem)
    Pci=[]
    Pci.append(tem)
    Pci.append(1-tem)
    oriEntropy = infoEntropy(Pci)                         #计算原始熵
    dataSet = list(set([data[j] for data in dataMat]))
    kind = len(dataSet)
    Pci = np.zeros(kind)
    count = np.zeros(kind)
    for d in range(kind):                                   #开始计算去feature后的信息熵
        for i in range(m):
            if dataMat[i][j] == dataSet[d]:
                Pci[d] += lables[i]                            #lables是0或1
                count[d]+=1
    Pci=Pci/count
    count=count/m
    Pcif = 1-Pci
    entropy = -Pci *np.log2(Pci)
    entropy += -Pcif *np.log2(Pcif)
    entropy = entropy*count
    entropy = sum(entropy)
    gain = oriEntropy-entropy                                 #计算信息增益
    return gain

a=[[1,0,0,1,1],[1,0,1,0,1],[0,1,1,0,0],[1,1,0,0,0],[1,1,0,1,1],[1,0,0,0,0]]
m = len(a)
j=2
n='买马匹'
print(infoGain(a,j,m,n))
