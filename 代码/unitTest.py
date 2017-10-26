'''
这里主要测试特征怎么取，单个块的准确率最高。
'''
import numpy as np
from ImageTool import towBitBP
import application as ap
import ImageTool as it
import learningTool as lt
import util as ut
import math
import sys

imagePath = "/home/junkai/桌面/数据/FaceOcc2/img"
X = 118
Y = 57
lenth = 82
width = 98
target = {}  # 记录一帧中目标位置，放在track里
target['x'] = X
target['y'] = Y
target['lenth'] = lenth
target['width'] = width
targetPosition = []  # 记录追踪过程中目标位置的序列
targetPosition.append(target)
numFeat = 4
numFern = 10
testX = 121
testY = 61
testLenth = 77
testWidth = 93

imaMat = it.image2Mat(imagePath + '/' + "0001.jpg", 1)
inteIma = it.getInteIma(imaMat)


posBag = it.getPosBag(X, Y, lenth, width)  # 正包
lables = list(np.ones(len(posBag)))  # 正包标签
negBag = it.getNegBag(X, Y, lenth, width, 1, 1)  # 负包
lables.extend(np.zeros(len(negBag)))  # 整个标签

offsetInfo = []  # 这个是块偏移信息，放外面
#print(allInstance)
blocksInfos = []        # 每个块的信息，这里一幅图就是一个块。
blockClassifier = []    # 每个块已训练好的的强分类器（adaboost分类器）
blockInfos = posBag.extend(negBag)
blocksInfos.append(blockInfos)  #为了使用以前写的randomFen函数，而这么搞的
randomFerns, dataMats, features = lt.randomFern(
    inteIma, blocksInfos, lables, numFeat, numFern)


