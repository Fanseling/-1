import numpy as np
from ImageTool import towBitBP           #此处彻底破坏了之前的低耦合想法
import application as app
def getPosition(path):                   #(文件操作)从给定的txt中，获取目标位置，注意，path是图片所在文件夹，要返回上一级
    i =len(path)-1
    while path[-1]!= '/':
        path=path[:i]
        i=i-1
    path=path[:i-1]
    path = path+"groundtruth_rect.txt"
    tar = []                                   #目标位置的列表，实在想不到单词了
    with open(path) as tarfile:
        targets = tarfile.readlines()
        for target in targets:
            tem = {}
            target.strip()
            locate = target.split(',')
            tem['x'] = locate[0]
            tem['y'] = locate[1]
            tem['lenth'] = locate[2]
            tem['width'] = locate[3]
            tar.append(tem)
    return tar

def randomSelect(lenth,width,num):                       #随机挑选y用于组成随机蕨的特征。参数为图像块的长宽，特征数量。在图像块上选取。
    x = 0                                                #这里是通用模块，固定以（0,0）为起点，特定示例用的时候加上该示例的起点x，y就行
    y = 0
    randX = 0                                                #随机X，随机Y，随机长宽。其实不用声明在外的。
    randY = 0
    randlen = 0
    randwid = 0
    randFeat=[]
    for i in range(num):
        randX = x+np.random.randint(0,lenth-2)              #-2的目的是防止选到长或宽上的最后一个像素了。
        randY = y+np.random.randint(0,width-2)
        lastlen = lenth + x - randX                        #计算以randX与randY开始的块的长宽上限，在纸上画一画就清楚了
        lastwid = width + y - randY
        randlen = np.random.randint(0,lastlen+1)
        tem = {}
        tem['x'] = randX
        tem['y'] = randY
        tem['lenth'] = lastlen
        tem['width'] = lastwid
        randFeat.append(tem)
    return randFeat

def data2Mat(inteIma,featurelist,blocksInfos):          #将图像矩阵，转换为2bitBP特征的特征值数组
    dataArray = []                                      #按照blocksInfos的一级顺序记录,这是个三维数组
    for blocksInfo in blocksInfos:
        blocksArr = []
        for  blockInfo in blocksInfo:
            values=[]
            for feature in featurelist:
                value = towBitBP(inteIma,blockInfo['x']+feature['x'],blockInfo['y']+feature['y'],feature['lenth'],feature['width'])
                values.append(value)
            blocksArr.append(values)
        dataArray.append(blocksArr)
    return dataArray

def basicOnBlock(dataArray，lables,numFeat):                         #以块为基准，每个块有个矩阵，矩阵是图片和这个快的特征值。  已测试
    dataMats = []
    n=len(dataArray[0])
    m=len(dataArray)                                       #n是块数，m是图数
    for i in range(n):                 #每一块
        dataMat=[]
        for j in range(m):             #每一张图片
            dataMat.append(dataArray[j][i].append(lables[j]))
        dataMats.append(dataMat)
    lables=[]
    for i in range(numFeat):           #重装lables
        lables.append(i)
    return dataMats,lables
    '''
    for i in range(len(dataArray)):
        for j in range(len(dataArray[i])):
            newLables.append(lables[i])
            dataMat.append(dataArray[i][j])
    #m,n=np.shape(dataMat)
    #for i in range(m):                                      #将lable加入到datamat的最后一列，方便后面建蕨
    #    dataMat[i].append(newLables[i])
    '''
    return np.mat(dataMat),newLables

def probability(lables,Pci):                               #概率计算
    Pi=[]
    count = 0
    for i in Pci:
        count += lables[i]
        p = count/len(Pci)
    Pi.append(p)
    Pi.append(1-p)
    return Pi

def  infoEntropy(Pci):                                  #计算信息熵，Pci是P(ci)，ci出现的概率，这里是个数组，i有多少数组有多少元素
    entropy = 0.0
    for P in Pci:
        entropy += P *np.log2(P)
    entropy=-entropy
    return entropy

def infoGain(dataMat,j):                              #计算给定特征的信息增益
    m,n=np.shape(dataMat)
    gain=0.0
    tem=dataMat[-1].sum()/m
    Pci.append(tem)
    Pci.append(1-tem)
    oriEntropy = infoEntropy(Pci)                             #计算原始熵
    Pc0 = [];Pc1 = [];Pc2 = [];Pc4 = []                       #已知共4种情况，不用排序。能不排序就不排序。
    for i in range(len(dataMat)):                                   #开始计算去feature后的信息熵
        if dataMat[i,j] == 0 : Pc0.append(i)
        elif dataMat[i,j] == 1 : Pc1.append(i)
        elif dataMat[i,j] == 2 : Pc2.append(i)
        elif dataMat[i,j] == 3 : Pc3.append(i)
    gain += len(Pc0)/m* infoEntropy(probability(lables,Pc0))
    gain += len(Pc1)/m* infoEntropy(probability(lables,Pc1))
    gain += len(Pc2)/m* infoEntropy(probability(lables,Pc2))
    gain += len(Pc3)/m* infoEntropy(probability(lables,Pc3))
    gain = gain-oriEntropy                                   #计算信息增益
    return gain

def featSelect(dataMat,lables):                          #建蕨时最大信息增益的特征 选择
    m,n=np.shape(dataMat)
    maxGain=-(np.inf)
    index= -1
    for i in lables:
        gain=infoGain(dataMat,i)
        if gain>=maxGain:
            maxGain=gain
            index = i
    return index

def devision(dataMat,i,value):                  #用于树的分裂，之前想的方法python3做不了
    devDataMat=[]
    for data in dataMat:
        if data[i]==value:
            temdata = data[:i]
            temdata.extend(data[i+1:])
            devDataMat.append(temdata)
    return devDataMat

def buildFerns(dataMat,lables):                      #这是以前《机器学习实战》的实现方法，效率真tm低，以后用自己想的改进下。
    m,n = np.shape(dataMat)
    result = [data[-1] for data in dataMat]
    if n==1:
        return dataMat.sum()/(float)m                #返回的是后验概率
    fern={}
    #dataMatTem=dataMat[:,:]
    #lablesTem = lables[:]
    featIndex = featSelect(dataMat,lables)
    lableJ = lables[featIndex]
    del lables[featIndex]
    fern[lableJ]={}
    elemset = [data[featIndex] for data in dataSet]
    elemset = set(elemset)                            #用来去重
    for elem in elemset:
        fern[lableJ][elem]={}                         #将dataMat中featIndex那一列的元素无重复的放入fern[lableJ]中。
        fern[lableJ][elem]=buildFerns(devision(dataMat,featIndex,elem),lables)
    return fern
    '''
    fern[index]={}
    childs=[]
    childLab=[]
    for i in range(4):                               #准备数据结构。子树的根有四个
        childs.append([])
        childLab.append([])
    for i in range(m):                               #遍历每一个块，将块按特征值分类，装进上面准备的结构中
        tem = dataMat[i,:]
        del tem[featIndex]
        childs[dataMat[i,featIndex]].append(tem)
        childLab[dataMat[i,featIndex]].append(lables[i])
    for i in range(4):                              #对每一个子树再递归本方法
        fern[index][i]=buildFerns(childs[i],childLab[i])
    return fern
    featureIndex = []                              #递归时只修改特征索引里的值，不动dataMat，大大减少计算量
    for i in range(n):
        featureIndex.append(i)
    del featureIndex[index]
    '''

def randomFern(inteIma,blocksInfos,lables,numFeat,numFern):                      #建成随机蕨组，强内聚，低耦合，这里直接调用上面的函数
    randomFerns = []                                            #八个块全部的蕨
    features = []
    for i in range(numFern):
        blocksFern = []                                         #每个块的蕨
        featurelist = randomSelect(blocksInfos[0][0]['lenth'],blocksInfos[0][0]['width'],numFeat)
        features.append(featurelist)
        dataArray= data2Mat(inteIma,featurelist,blocksInfos))             #这是个三维数组
        dataMats,lables= basicOnBlock(dataArray,lables,numFeat)           #改变了dataMat和lables的结构，具体看函数注释
        #将block和picture的维度换一下，现在block是第一维,并且将lables并入dataMats最后一列，lables重装特征序列号
        for dataMat in dataMats:                                #对于每个block。这样，每个dataMat就可以当成普通分类树来写了
            fern={}                                             #一个蕨
            dataMat=np.mat(dataMat)
            fern=buildFerns(dataMat,lables)
            blocksFern.append(fern)
        randomFerns.append(blocksFern)
        return randomFerns,dataMats, features

'''
    由于onlineAdaboost包含Adaboost全部代码，事实上这里只有onlineAdaboost。此处的
onlineBoosting应该说只能更新弱分类器的权值，而不能替换整个弱分类器，因为在2bitBP上弱分类器茫
茫多，不能保证新生成的弱分类器比原来的好。
    新想法：可以先生成个弱分类器池M，替换原强分类器里的弱分类器为池M中最优的。但不能保证len(M)为
多少时精度较好,可能需要大大提高计算量。这个想法在下面实现。
'''

def bestFern(randomFerns,dataMat,lables,D):
    minErr = np.inf
    bestIndex = -np.inf
    for index in range(len(randomFerns)):                           #对于每个块的蕨
        result=[]
        result = app.fernClassify(randomFerns[index], dataMat)      #err为错误率，未正确分类样本数/总样本数
        err=abs(result-lables).sum()/len(lables)
        if err<minErr:
            minErr = err
            bestIndex = index
    return index, minErr

def refrashD(alpha,D):
    for i in range(len(D)):
        D[i, 0]=(D[i, 0]*np.e**-alpha)/D.sum()
    return D

def AdaBoost(randomFerns,dataMat,num):            #此处的randomFerns是单个块的，num是这个强分类器要多少弱分类器
    classifiers = []
    lables = np.mat([[data[-1] for data in dataMat]]).T     #注意，最后一行是标签项,已矩阵化,列向量矩阵
    for data in dataMat:
        del data[-1]
    dataMat=np.mat(dataMat)
    m,n=np.shape(dataMat)
    D = mat(ones((m,1))/m)                     #初始化样本权重向量为1/m,注意，这是个列向量
    for i in range(num):                       #对于每个弱分类器
        classifier={}
        bestFernIndex,err=bestFern(randomFerns, dataMat, lables, D)
        alpha = 0.5*log((1-err)/err)           #计算alpha
        D = refrashD(alpha,D)                  #更新权重矩阵D
        classifier["fern"]=randomFerns[bestFernIndex]  # 此处的classifier是若分类器
        classifier["alpha"] = alpha
        classifiers.append(classifier)         #classifiers是强分类器了
    return classifiers
