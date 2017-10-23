import numpy as np
from ImageTool import towBitBP  # 此处彻底破坏了之前的低耦合想法
import application as ap
import ImageTool as it
import learningTool as lt
import util as ut
import math
import sys

imagePath = "/home/junkai/桌面/数据/FaceOcc2/img"
X=118
Y=57
lenth=82
width=98
target = {}                                       #记录一帧中目标位置，放在track里
target['x'] = X
target['y'] = Y
target['lenth'] = lenth
target['width'] = width
targetPosition = []                                        #记录追踪过程中目标位置的序列
targetPosition.append(target)
numFeat= 4
numFern = 12
testX=121
testY=61
testLenth=77
testWidth=93

imaMat = it.image2Mat(imagePath + '/' + "0001.jpg", 1)
inteIma = it.getInteIma(imaMat)


posBag = it.getPosBag(X,Y,lenth,width)                 #正包
lables =list(np.ones(len(posBag)))                       #正包标签
negBag = it.getNegBag(X,Y,lenth,width,2,4)                 #负包
lables.extend(np.zeros(len(negBag)))                #整个标签
#负包并不是以整个图片出现的，而是以同一个块的组合出现，使每个块都学习到相同的

offsetInfo=[]#这个是块偏移信息，放外面
#print(allInstance)
blocksInfos = []
blockClassifier = []                              # 每个块已训练好的的强分类器（adaboost分类器）

for instance in posBag:                #对于每个正示例分块并存储。这个循环只能干这么点事。
    # 分块,第一帧就用target了，没用targetPosition[-1]
    blocksInfo, offsetInfo = it.imageFrag(
        instance['x'], instance['y'], instance['lenth'], instance['width'], 2, 4)
    #blocksInfo记录块的起始位置xy以及块的长宽。offsetInfo记录块相对于图像的偏移信息
    blocksInfos.append(blocksInfo)    #所有示例的偏移信息是一样的，不需要在添加一个列表

for i in range(len(negBag)):                           #负示例分块,每个示例八个块，每个块内容是一模一样
    blocksInfo = []
    for j in range(8):
        blocksInfo.append(negBag[i])
    blocksInfos.append(blocksInfo)

#**********************测试*********************
'''
tems=[]
blocksInfoTem, offsetInfoTem = it.imageFrag(
    targetPosition[-1]['x'], targetPosition[-1]['y'], targetPosition[-1]['lenth'], targetPosition[-1]['width'], 2, 4)

#imaMatTem=it.mark(imaMat,blocksInfoTem[0]['x'],blocksInfoTem[0]['y'],blocksInfoTem[0]['lenth'],blocksInfoTem[0]['width'])


for i in range(len(blocksInfoTem)):
    Tem = {}
    Tem['position']=blocksInfoTem[i]
    Tem['blockIndex'] = i
    tems.append(Tem)

targetTem = it.objectConfirm(targetPosition[-1]['x'], targetPosition[-1]['y'],tems,offsetInfoTem)
print(targetTem)
imaMatTem=it.mark(imaMat,targetPosition[-1])
imaMatTem=it.mark(imaMat,tems[0]['position'])
imaMatTem=it.mark(imaMat,tems[1]['position'])
imaMatTem=it.mark(imaMat,tems[2]['position'])
imaMatTem=it.mark(imaMat,tems[3]['position'])
imaMatTem=it.mark(imaMat,targetTem)
it.showIma(imaMatTem)
sys.exit("测试退出")
'''
#**********************结束***********************

#循环结束，数据准备结束，下面开始学习

randomFerns,dataMats,features = lt.randomFern(inteIma,blocksInfos,lables,numFeat,numFern)
#dataMats第一维是蕨，第二维是块，第三维是示例，第四维是每个蕨的特征值序列
#randomFerns第一维是建立的数个蕨，第二维是每个块。
#print(randomFerns)
fn = len(randomFerns[0])
for i in range(fn):                        #对每个块建立分类器
    randomfern = [x[i] for x in randomFerns]         #真是想不到什么名字了,并且randomFerns[:][i]的方式不行！
    classifier=[]                          #强分类器
    classifier = lt.AdaBoost(randomfern, [x[i] for x in dataMats], 8)
    blockClassifier.append(classifier)
temData=[]
for dataMat in dataMats :
    temData.append([x[0] for x in dataMat])       #此处测试通过，意义是取第零个示例，示例在第三维
block = ut.blockSortedByP(blockClassifier, temData)
blockSorted = [x[0] for x in block]
#print(blockSorted)

#print(len(randomFerns))
#print(randomFerns.type())
#print(randomFerns)

#*******************第二帧*********************
#*******************第二帧*********************
#*******************第二帧*********************
#*******************第二帧*********************
#*******************第二帧*********************
imaMat = it.image2Mat(imagePath + '/' + "0016.jpg", 1)
inteIma = it.getInteIma(imaMat)
blocksInfo = []                           #检测出的块的位置,可以放在这里
blocks=[]                                 #记录P>0.5的最开始四个快的位置信息
obscuredBlock = []                        #被遮挡的块，每帧重置
m,n=np.shape(inteIma)
n1=max(int(targetPosition[-1]['x']-targetPosition[-1]['lenth']*0.3),0)
n2 = min(int(targetPosition[-1]['x']+targetPosition[-1]['lenth']*1.3),n-2)
m1 = max(int(targetPosition[-1]['y']-targetPosition[-1]['width']*0.3),0)
m2=min(int(targetPosition[-1]['y']+targetPosition[-1]['width']*1.3),m-2)
getRange = [n1,n2,m1,m2]   #取值范围，长（min，max+1），宽（min，max+1）,后面用range()所以加1。
dataMats, dataPosition = ut.getData(inteIma, blocksInfos[0][0],getRange,features) #dataMats是list结构
#此处dataMat第一维是每个块（此处应理解为示例），第二维是每个随机蕨，第三维是蕨内容
#print(dataMats)

for i in blockSorted:            # 此处应该没错，就是 blockSorted，计算每个块的概率
    probability=[]                            #对下一帧进行检测，各个块为目标块的概率向量
    for dataMat in dataMats:
        #使用某个块的分类器对数据分析，得到概率向量,此处dataMats与上面不同，是二维数组
        probability.append(ap.adaBoostClassify(blockClassifier[i],dataMat))  #使用某个块的分类器对数据分析，得到概率向量
        #此处dataMat第一维是每个块（此处应理解为示例），第二维是每个随机蕨，第三维是蕨内容

    maxP = max(probability)
    #print(probability)
    print(i,'处最大概率为',maxP)
    #maxIndex = probability.index(maxP)

    maxIndex = ut.getList(probability,maxP)    #得到哪几个块是概率最大的块
    print("maxIndex出现次数：",len(maxIndex))
    maxIndex = maxIndex[int(len(maxIndex)/2)]
    #print("maxIndex最终位置",maxIndex)
    if maxP>0.6:
        block={}
        block['position'] =dataPosition[maxIndex]
        block['blockIndex'] = i
        blocks.append(block)
        #print("检测的块数",i,"最大概率为：",maxP)
        #print('位置：     ',dataPosition[maxIndex])
        #print('标准位置为：',blocksInfos[0][i])
    if len(blocks) == 4: break
if len(blocks)<4 :
    sys.exit("第"+image+"张图片遮挡或变化过多，检测失败")

target = it.objectConfirm(targetPosition[-1]['x'], targetPosition[-1]['y'], blocks, offsetInfo)
print("检测到的目标：",target)
print("原始目标位置：",targetPosition[-1])
#print("真实目标位置：{'x': 121, 'y': 61, 'lenth': 77, 'width': 93,}",)


#for i in range(76846,89040):
#    imaMatTem=it.mark(imaMat,dataPosition[i])
imaMatTem=it.mark(imaMat,blocks[0]['position'])
imaMatTem=it.mark(imaMat,blocks[1]['position'])
imaMatTem=it.mark(imaMat,blocks[2]['position'])
imaMatTem=it.mark(imaMat,blocks[3]['position'])
imaMatTem=it.mark(imaMat,target)

it.showIma(imaMatTem)
sys.exit("断点退出。")









targetPosition.append(target)  # 记录追踪位置

#*************计算检测出的目标位置的中心误差***************
targetX,targetY = it.strat2center(target["x"], target["y"], target["lenth"], target["width"])
finalX, finalY = it.strat2center(
    StanPosition["x"], StanPosition["y"], StanPosition["lenth"], StanPosition["width"])
centerErr.append(math.sqrt((targetX - finalX)**2 + (targetY - finalY)**2))
