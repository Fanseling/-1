#这里是程序的主流程
import os
import ImageTool as it                             #包含图像操作相关函数
import learningTool as lt                          #其他工具函数基本都在里面，包括目标位置的提取
import numpy as np
import util as ut
import application as ap
import time

#一下一段调整参数
version = 1                                       #使用论文算法
numFeat                                          #每个蕨用多少个2bitBP特征
numFern                                          #随机蕨池要多少个
num                                              #adboosting要组多少个随机蕨
#以上是调整参数

imagePath = input("输入图片序列的路径：")                      #初始化
color = input("输入图片颜色位数（1表示黑白，3表示彩色）：")
images = os.listdir(imagePath).sort()
target = {}                                       #记录一帧中目标位置，放在track里
X,Y,lenth,width = map(int,input ("输入目标初始坐标,及长宽,空格隔开：").split())   #第一帧手动输入
target['x'] = X
target['y'] = Y
target['lenth'] = lenth
target['width'] = width

#以下是循环和循环要用的变量的初始化了
StanPosition =ut.getPosition(path)                #从标准位置文件中获取目标真实位置
CenterError =[]
blockSorted = []                                  # 块按概率排序的列表，八个块都在里面
targetPosition = []                                        #记录追踪过程中目标位置的序列
targetPosition.append(target)
blockClassifier = []                              # 每个块已训练好的的强分类器（adaboost分类器）
blocksInfos = []                                  # 维护示例的分块信息
offsetInfos = []                                  # 维护分块的偏移信息
features = []

for image in images:                              #对于每一帧
    imaMat = it.image2Mat(imagePath+'/'+image,color)  #图像转矩阵
    inteIma = it.getInteIma(imaMat)                   #积分图
    if image == '0001.jpg':                       #第一帧只学习，不分类，单独拿出来
        originIndex = 0
        posBag = getPosBag(X,Y,lenth,width)                 #正包
        lables = np.ones(len(posBag))                       #正包标签
        negBag = getNegBag(X,Y,lenth,width)                 #负包
        lables.extend(np.zeros(len(negBag)))                #整个标签
        allInstance = posBag.extend(negBag)                 #整个示例样本，allInstance只是新建了个指针

        #上面两个都是二位列表，blocksInfos[0][0][x]是第一个示例的第一个块的起始坐标的x值

        for instance in allInstance:                #对于每个示例分块并存储。这个循环只能干这么点事
            # 分块,第一帧就用target了，没用targetPosition[-1]
            blocksInfo, offsetInfo = imageFrag(
                image, target['x'], target['y'], target['lenth'], target['width'], 2, 4)
            #blocksInfo记录块的起始位置xy以及块的长宽。offsetInfo记录块相对于图像的偏移信息
            blocksInfos.append(blocksInfo)
            offsetInfos.append(offsetInfo)
        #循环结束，数据准备结束，下面开始学习
        randomFerns,dataMats,features = lt.randomFern(inteIma,blocksInfos,lables,numFeat，numFern)         #所有块都拿去学习建蕨，但只有4个块用来检测。numFeat是每块选择的特征数量，暂没定是多少
        #dataMats第一维是块，第二维是示例，第三维是特征值序列
        #randomFerns第一维是建立的数个蕨，第二维是每个块。
        #numFeat是每块选择的特征数量，暂没定是多少
        #numFern是建多少个随机蕨以供boosting选择
        #features是在每个块的哪个位置建立的特征值，这个是每帧用一次的。所有的块都一样这是个二维数组，第一维是每个随机蕨，第二位是随机蕨中每个特征值的位置
        #features是有问题的，第二维并未随建蕨而改变顺序。新发现，不需要随建蕨而改变顺序，每次都要重新建蕨，以学习图像的变化,计算量飙升。

        fn = len(randomFerns[0])
        for i in range(fn):                        #对每个块建立分类器
            randomfern = [x[i] for x in randomFerns]         #真是想不到什么名字了,并且randomFerns[:][i]的方式不行！
            classifier=[]                          #强分类器
            classifier = lt.AdaBoost(randomfern,dataMats[i],num)    #**********num是多少还没定***************
            blockClassifier.append(classifier)

        # 此处的dataMats还是个list，只取初始示例的八个块的特征值，去计算
        block = ut.blockSortedByP(blockClassifier, [x[0] for x in dataMats])
        blockSorted = [x[1] for x in block]


    else:                                     #第二帧及以后
        now = time.time()                     #现在时间（以秒为单位）
        blocksInfo = []                           #检测出的块的位置
        if version == 1 :                      #别人的方法，即没有轨迹预测和全块学习
            #以块为单位进行全图片检测，四个块各检测一个滑动窗口‘遍’，找出概率超过50%并且最高的，作为预测点。
            dataMats,dataPosition = ut.getData(inteIma)   #没写完！应用滑动窗口，获取数据.此处滑动串口不改变图像，而是改变检测窗口大小，改两个模块
            for i in blockSorted:
                probability = ap.adaBoostClassify(blockClassifier[i],dataMats)  #使用某个块的分类器对数据分析，得到概率向量
                maxP = probability.max()
#                print(maxP)
                maxIndex = list(probability).index(maxP)    #得到第几个块是检测到的块
                if maxP>0.5: blocks.append(dataPosition[maxIndex])
                if len(blocks) == 4: break      
            if len(blocks)<4 :
                print ("第"+image+"张图片遮挡或变化过多，检测失败")
                continue
            target = it.objectConfirm(targetPosition[-1]['x'], targetPosition[-1]['y'], blocks, offsetInfo) 
            targetPosition.append(target)
            #检测结束,开始计算哪里有遮挡，（概率小于50%认为有遮挡）
            blocksInfo, offsetInfo = imageFrag(image, targetPosition[-1]['x'], targetPosition[-1]['y'],
                      targetPosition[-1]['lenth'], targetPosition[-1]['width'], 2, 4)

            #遮挡计算结束，开始对新信息学习，大部分是重复if里的代码。注意，现在还没有计算错误
            posBag = getPosBag(targetPosition[-1]['x'], targetPosition[-1]['y'],
                               targetPosition[-1]['lenth'], targetPosition[-1]['width'])  # 正包
            lables = np.ones(len(posBag))                                                 # 正包标签
            negBag = getNegBag(targetPosition[-1]['x'], targetPosition[-1]['y'],
                               targetPosition[-1]['lenth'], targetPosition[-1]['width'])  # 负包
            lables.extend(np.zeros(len(negBag)))                                          # 整个标签
            allInstance = posBag.extend(negBag)                                           # 整个示例样本，allInstance只是新建了个指针
            for instance in allInstance:  # 对于每个示例分块并存储。这个循环只能干这么点事
                blocksInfo, offsetInfo = imageFrag(
                    image, targetPosition[-1]['x'], targetPosition[-1]['y'],
                    targetPosition[-1]['lenth'], targetPosition[-1]['width'], 2, 4)      # 分块
                #blocksInfo记录块的起始位置xy以及块的长宽。offsetInfo记录块相对于图像的偏移信息
                blocksInfos.append(blocksInfo)
                offsetInfos.append(offsetInfo)
        #循环结束，数据准备结束，下面开始学习
        # 所有块都拿去学习建蕨，但只有4个块用来检测。numFeat是每块选择的特征数量，暂没定是多少
        randomFerns, dataMats = lt.updateFern(inteIma, blocksInfos, lables, features,numFeat，numFern)
        fn = len(randomFerns[0])
        for i in range(fn):  # 对每个块更新分类器
            # 真是想不到什么名字了,并且randomFerns[:][i]的方式不行！
            randomfern = [x[i] for x in randomFerns]
            classifier = []                                          # 强分类器
            # **********num是多少还没定***************
            classifier = lt.AdaBoost(randomfern, dataMats[i], num)
            blockClassifier.append(classifier)

        # 此处的dataMats还是个list，只取初始示例的八个块的特征值，去计算
        block = ut.blockSortedByP(blockClassifier, [x[0] for x in dataMats])
        blockSorted = [x[1] for x in block]
