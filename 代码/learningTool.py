import numpy as np
from ImageTool import towBitBP           #此处彻底破坏了之前的低耦合想法
import application as app
import math
import random

def randomSelect(lenth, width, num):  # 随机挑选y用于组成随机蕨的特征。参数为图像块的长宽，特征数量。在图像块上选取。已测试
    randFeat = []
    i = 0
    while(i<num):
        randLen = random.randint(30,80) #20-30能看，30-40能看，40-50能看，但是都有些块混淆/ 50-60块分开了，但是概率为1
        randWid = random.randint(30,70) #的块越来越多，y轴的不准确开始变多，从40-50开始。60-70开始有些不能让人满意了，下降了
                                        #还是能看。
        tem = {}
        #lastX = math.ceil(2 / lenth)  # 两个像素占长的百分之多少
        #lastY = math.ceil(2 / width)
        # 由于图像从第0个像素开始，所以是0-100。计算具体像素点位置是长宽记得减1
        randX = np.random.randint(0, 100 - randLen)
        randY = np.random.randint(0, 100 - randLen)
        #lastLen = 100 - randX
        #lastWid = 100 - randY
        #randLen = np.random.randint(lastX, lastLen)  # lastX....好好想想吧
        #randWid = np.random.randint(lastY, lastWid)
        tem['x'] = randX
        tem['y'] = randY
        tem['lenth'] = randLen
        tem['width'] = randWid
        randFeat.append(tem)
        i=i+1
    return randFeat   #即featurelist

def getFeature(featurelist, blockInfo):  # 这里是通用模块，固定以（0, 0）为起点，特定示例用的时候加上该示例的起点x，y就行
    result = []
    #print(featurelist)
    #print()
    #print(blockInfo)
    for feature in featurelist:
        tem={}
        tem['x'] = round(feature['x']/100 * (blockInfo['lenth'])) #此处减一严格来说是有问题的，但影响不大。
        tem['y'] = round(feature['y'] /100 * (blockInfo['width']))
        tem['lenth'] = round(feature['lenth']/100 * (blockInfo['lenth']))
        tem['width'] = round(feature['width'] /100 * (blockInfo['width']))
        result.append(tem)
    return result

'''
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
'''

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

def basicOnBlock(dataArray,lables,numFeat):  #以块为基准，每个块有数个矩阵，矩阵是图片和这个快的特征值。  已测试
    dataMats = []                      #dataMats第一维是块，第二维是示例，第三维是特征值序列（数个蕨的2bitbp值）
    n=len(dataArray[0])
    m=len(dataArray)                   #n是块数，m是图数
    #print(m)
    for i in range(n):                 #每一块
        dataMat=[]
        for j in range(m):             #每一张图片


            #可能有问题
            #可能有问题
            #可能有问题
        #可能有问题
            dataArray[j][i].append(lables[j])
            dataMat.append(dataArray[j][i])
            #print(dataMat)
        dataMats.append(dataMat)
        #print(dataMats)
    lables=[]
    for i in range(numFeat):           #重装lables
        lables.append(i)
    #print(lables)
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
    return dataMat,newLables

def probability(lables,dataIndex):                               #概率计算
    Pi=[]
    count = 0
    for i in dataIndex:
        count += lables[i]
    if len(dataIndex)==0:
        Pi.append(0)
        Pi.append(0)
    else:
        p = count/len(dataIndex)
        Pi.append(p)
        Pi.append(1-p)
    return Pi

def  infoEntropy(Pci):                                  #计算信息熵，Pci是P(ci)，ci出现的概率，这里是个数组，i有多少数组有多少元素
    entropy = 0.0
    for P in Pci:
        temp= -P *np.log2(P)
        if temp != temp : temp=0
        entropy +=temp
    return entropy

def infoGain(dataMat,j,m,n):                              #计算给定特征的信息增益
    contentErr = 0
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
    #print("dataMat",np.mat(dataMat))
    #print("dataSet",j,dataSet)

    kind = len(dataSet)
    Pci = np.zeros(kind)
    count = np.zeros(kind)
    for d in range(kind):                                   #开始计算去feature后的信息熵
        for i in range(m):
            if dataMat[i][j] == dataSet[d]:
                Pci[d] += lables[i]                            #lables是0或1
                count[d]+=1
    #print("count是:",count)
    #print("Pci是：",Pci)
    Pci=Pci/count
    count=count/m
    Pcif = 1-Pci
    entropy = -Pci *np.log2(Pci)
    for i in range(len(entropy)):                         #移除np.nan
        if entropy[i] != entropy[i]:
            entropy[i] = 0

    temp = -Pcif *np.log2(Pcif)
    for i in range(len(temp)):                         #移除np.nan
        if temp[i] != temp[i]:
            temp[i] = 0
    entropy += temp
    #print("entropy是：",entropy)
    if np.nan in entropy : entropy[entropy.index(np.nan)]=0
    entropy = entropy*count
    entropy = sum(entropy)
    gain = oriEntropy-entropy                                 #计算信息增益
    return gain

def featSelect(dataMat,lables,m,n):                          #建蕨时最大信息增益的特征 选择
    if len(lables)==1:
        #print("最终选择特征(唯一选择)：",lables[0])
        return lables[0]
    maxGain=-(np.inf)
    index= -np.inf
    #print(lables)
    #print("featSelect")
    #print(dataMat)
    for i in lables:
        gain=infoGain(dataMat,i,m,n)
        #print("信息增益",gain)
        if gain>=maxGain:
            maxGain=gain
            index = i
    #print("最终选择特征：",index)
    return index

def devision(dataMat,i,value):                  #用于树的分裂，之前想的方法python3做不了
    #print(i)
    devDataMat=[]
    #dataMat = np.array(dataMat)
    for data in dataMat:
        #print("data is:",data)
        if data[i]==value:
            temdata = data[:i]
            #print("temdata is:",temdata)
            temdata.extend(data[i+1:])
            devDataMat.append(temdata)
    return devDataMat

def buildFerns(dataMat, lables):  # 这是以前《机器学习实战》的实现方法，效率真tm低，以后用自己想的改进下。
    m = len(dataMat)
    n = len(dataMat[0])
    #print("lables是：",lables)
    result = [data[-1] for data in dataMat]
    if len(set(result))==1:
        #print("剩下的样本为：",dataMat)
        #print("剩下所有样本标签都为：",set(result),result[0])
        return result[0]
    if len(lables)==0:
        #print("一处后验概率为：",sum(result)/float(m))
        return sum(result)/float(m)                #返回的是后验概率
    #print(np.mat(dataMat))
    #print(lables)
    fern={}
    #dataMatTem=dataMat[:,:]
    #lablesTem = lables[:]
    #print("buildFerns")
    #print(dataMat)
    featIndex = featSelect(dataMat,lables,m,n)
    #print("选择的特征序号",featIndex)
    lableJ =featIndex
    index = lables.index(featIndex)#实际上本蕨的featIndex
    #print("本蕨的位置：",index)
    fern={lableJ:{}}
    #print(fern)
    elemset = [data[index] for data in dataMat]
    del lables[index]
    #print(elemset)
    elemset = set(elemset)                            #用来去重
    for elem in elemset:
        labtem = lables[:]                            #复制个副本给下一个迭代
        fern[lableJ][elem]={}                         #将dataMat中featIndex那一列的元素无重复的放入fern[lableJ]中。
        fern[lableJ][elem]=buildFerns(devision(dataMat,index,elem),labtem)
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


# 建成随机蕨，numFeat是每个蕨里特征数量，numFern是每个块蕨的数量（用来建boosting）。
def randomFern(inteIma, blocksInfos, lables, numFeat, numFern):
    randomFerns = []                                            #八个块全部的蕨
    features = []                                               #存储每个块特征的信息
    datasMats=[]                                                #遗留问题
    for i in range(numFern):                                    #对于每个随机蕨（要建立数个蕨来组adaboost）
        blocksFern = []                                         #每个块的蕨
        featurelist = randomSelect(blocksInfos[0][0]['lenth'],blocksInfos[0][0]['width'],numFeat)
        features.append(featurelist)                            #此处保存的是相对位置，需要转换计算
        featurelist = getFeature(featurelist, blocksInfos[0][0])   #每个块用同一个特征列表，所以只需要一个block的信息就足够了
        #print(featurelist)
        dataArray= data2Mat(inteIma,featurelist,blocksInfos)             #这是个三维数组
        #print(dataArray[0])  #dataArray是个三维数组，第一维示例，第二维块，第三维特征值序列
        dataMats,lable= basicOnBlock(dataArray,lables,numFeat)           #改变了dataMat和lables的结构，具体看函数注释
        #print(dataMats[0])
        #将block和picture的维度换一下，现在block是第一维,并且将lables并入dataMats最后一列，lables重装特征序列号
        for dataMat in dataMats:                                #对于每个block。这样，每个dataMat就可以当成普通分类树来写了
            fern={}                                             #一个蕨
            #dataMat=np.mat(dataMat)
            #print("randomFern")
            #print(dataMat)
            dateMatTem = dataMat[:]
            lablesTem = lable[:]
            fern=buildFerns(dateMatTem, lablesTem)
            #print(fern)
            blocksFern.append(fern)
        #print(blocksFern)
        randomFerns.append(blocksFern)
        datasMats.append(dataMats)
        #print("完成第",i+1,"个随机蕨")
    return randomFerns,datasMats, features                   #返回的dataMates最后一行是标签

def updateFern(inteIma, blocksInfos, lables, features, numFeat, numFern, obscuredBlock):
    randomFerns = []                                            #八个块全部的蕨
    features = []                                               #存储每个块特征的信息
    datasMats=[]                                                #遗留问题
    for i in range(numFern):
        blocksFern = []                                         #每个块的蕨
        features = getFeature(features[i], blocksInfos[0][0])
        dataArray= data2Mat(inteIma,features,blocksInfos)             #这是个三维数组
        dataMats,lables= basicOnBlock(dataArray,lables,numFeat)           #改变了dataMat和lables的结构，具体看函数注释
        #将block和picture的维度换一下，现在block是第一维,并且将lables并入dataMats最后一列，lables重装特征序列号
        for j in range(len(dataMats)):                                #对于每个block。这样，每个dataMat就可以当成普通分类树来写了
            if j in obscuredBlock:continue                      #跳过被覆盖的块
            dataMat = dataMats[j]
            fern={}                                             #一个蕨
            dataMat=np.mat(dataMat)
            fern=buildFerns(dataMat, lables)
            blocksFern.append(fern)
        randomFerns.append(blocksFern)
        datasMats.append(dataMats)
    return randomFerns,datasMats                             #返回的dataMates最后一行是标签

'''
    由于onlineAdaboost包含Adaboost全部代码，事实上这里只有onlineAdaboost。此处的
onlineBoosting应该说只能更新弱分类器的权值，而不能替换整个弱分类器，因为在2bitBP上弱分类器茫
茫多，不能保证新生成的弱分类器比原来的好。
    新想法：可以先生成个弱分类器池M，替换原强分类器里的弱分类器为池M中最优的。但不能保证len(M)为
多少时精度较好,可能需要大大提高计算量。这个想法在下面实现。
'''

def bestFern(randomFerns,dataMat,lables,D):
#dataMat第一维是数个蕨，第二维是示例，第三维是特征序列
#randomFern是用于组adaboost的数个随机蕨，一维
    minErr = 1
    bestIndex = -np.inf
    bestresult=[]
    for index in range(len(randomFerns)):                           #对于每个蕨(9月实验是10个)
        #print(index)

        result=[]
        result = app.fernClassify(randomFerns[index], dataMat[index])      #返回的结果是各样本为1的概率
        #print(result)
        for i in range(len(result)):                                #将分类概率转化为标签
            if result[i]>0.5 : result[i] = 1.0
            else: result[i] = 0.0
        #print(sum(abs(np.array(result)-np.array(lables))))
        #print(float(len(lables)))
        err=np.mat([abs(np.array(result)-np.array(lables))])*D                 #err为错误率，未正确分类样本数/总样本数
        #print("err:",err)
        #print(D)
        if err<minErr:
            minErr = err
            #print("minErr:",minErr)
            bestIndex = index
            bestresult = result.copy()
    #print("一次最优fern选择结束")
    return bestIndex, minErr,bestresult

def refrashD(alpha,D):
    for i in range(len(D)):
        D[i,0]=(D[i,0]*np.e**-alpha)/D.sum()
    return D

# 此处的randomFerns是单个块的，num是这个强分类器要多少弱分类器
def AdaBoost(randomFerns, dataMat, num):
    #dataMat第一维是数个蕨，第二维是示例，第三维是特征序列
    #randomFern是用于组adaboost的数个随机蕨，一维
    lablesTem = []
    classifiers = []
    lables = [data[-1] for data in dataMat[0]]     #注意，最后一行是标签项,已矩阵化,列向量矩阵
    for i in range(len(lables)):
        if lables[i]==0.0:
            lablesTem.append(-1.0)
        else :
            lablesTem.append(1.0)


    for data in dataMat:                      #删除标签位。
        for x in data:
            del x[-1]
    m,n=np.shape(np.mat(dataMat[0]))
    D = np.mat(np.ones((m,1))/m)                     #初始化样本权重向量为1/m,注意，这是个列向量

    for i in range(num):                       #对于每个要组adaboost的弱分类器，不是全部弱分类器
        classifier={}
        bestFernIndex,err,bestresult=bestFern(randomFerns, dataMat, lables, D) #选择当前最好的随机蕨
        #print("选择的若分类器是：",bestFernIndex)
        #print("err是：",err)

        alpha = 0.5*math.log(((1-err)/max(err,1e-16)),math.e)           #计算alpha
        #print("alpha是：",alpha)

        for i in range(len(bestresult)):
            if bestresult[i]==0:bestresult[i]=-1.0

        '''
        expon = np.multiply(-1*alpha*np.mat(lables).T,np.mat(bestresult).T) #exponent for D calc, getting messy
        D1 = np.multiply(D,np.exp(expon))                              #Calc New D for next iteration
        D = D1/D1.sum()

        '''
        #print(bestresult)
        tem=-1*alpha*np.multiply(np.mat(lablesTem).T,np.mat(bestresult).T)     #更新权重矩阵D
        D1=np.multiply(D,np.exp(tem))
        D2 = D1.sum()
        D=D1/D2                                       #此处是《深入理解...》给的算法
        #print(tem)
        #D=refrashD(alpha,D)
        #print(D)
        classifier["fern"]=randomFerns[bestFernIndex]  # 此处的classifier是若分类器
        classifier["alpha"] = alpha
        #print("alpha:",alpha)
        classifier["index"] = bestFernIndex
        #print("index:",bestFernIndex)
        classifiers.append(classifier)         #classifiers是强分类器了
    #print("一个adaboosting组建完成")
    return classifiers
