import os
import ImageTool as it                             #包含图像操作相关函数
import learningTool as lt                          #其他工具函数基本都在里面，包括目标位置的提取
import numpy as np
imagePath = input("输入图片序列的路径：")                      #初始化
color = input("输入图片颜色位数（1表示黑白，3表示彩色）：")
images = os.listdir(imagePath).sort()
m = len(images)
track = []                                        #记录追踪过程中目标位置的序列
target = {}                                       #记录一帧中目标位置，放在track里
X,Y,lenth,width = map(int,input ("输入目标初始坐标,及长宽,空格隔开：").split())
target[x] = X
target[y] = Y
target[lenth] = lenth
target[width] = width
track.append(target)
StanPosition =lt.getPosition(path)                #从标准位置文件中获取目标真实位置
CenterError =[]
for image in images:                              #对于每一帧
    inteIma = getInteIma(image)                   #积分图
    if image == '0001.jpg':                       #第一帧只学习，不分类，单独拿出来
        imaMat = it.image2Mat(imagePath+'/'+image)
        posBag = getPosBag(X,Y,lenth,width)                 #正包
        lables = np.ones(len(posBag))                       #正包标签
        negBag = getNegBag(X,Y,lenth,width)                 #负包
        lables.extend(np.zeros(len(negBag)))                #整个标签
        allInstance = posBag.extend(negBag)                 #整个示例样本，allInstance只是新建了个指针
        blocksInfos=[]                                      #维护所有示例的分块信息
        offsetInfos=[]
        #上面两个都是二位列表，blocksInfos[0][0][x]是第一个示例的第一个块的起始坐标的x值

        for instance in allInstance:                #对于每个示例分块并存储。这个循环只能干这么点事
            blocksInfo,offsetInfo = imageFrag(image,instance[x],instance[y],target[lenth],target[y],2,4)   #分块
            #blocksInfo记录块的起始位置xy以及块的长宽。offsetInfo记录块相对于图像的偏移信息
            blocksInfos.append(blocksInfo)
            offsetInfos.append(offsetInfo)
        #循环结束
        randomFerns = lt.randomFern(inteIma,blocksInfos,lables,numFeat，numFern)         #所有块都拿去学习建蕨，但只有4个块用来检测。numFeat是每块选择的特征数量，暂没定是多少
        #randomFerns第一维是建立的数个蕨，第二维是每个块。

        fn = len(randomFerns[0])
        for i in range(fn):                    #对每个块建立分类器
            randomfern = randomFerns[:][i]     #真是想不到什么名字了
            AdaBoost(randomfern)



    else:                                     #第二帧及以后
