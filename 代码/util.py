'''
通用工具模块
'''
import application as ap
import learningTool as lt
import ImageTool as it

def getPosition(path):                   #(文件操作)从给定的txt中，获取目标位置，注意，path是图片所在文件夹，要返回上一级
    path = path+"groundtruth_rect.txt"   #已测试
    tar = []                                   #目标位置的列表，实在想不到单词了
    with open(path) as tarfile:
        targets = tarfile.readlines()
        for target in targets:
            tem = {}
            target = target.strip()
            locate = target.split(',')
            tem['x'] = locate[0]
            tem['y'] = locate[1]
            tem['lenth'] = locate[2]
            tem['width'] = locate[3]
            tar.append(tem)
    return tar

def blockSortedByP(blockClassifier,dataMat):          #已测试，根据每个块在学习到的分类器上的应用，得出概率，根据概率排序
    #dataMat第一维是蕨，第二维是块，第三维是每个蕨的特征值序列。这是第0个示例的dataMat
    #blockClassifier应该看做一维数组，虽然它是二维的。这是每个块的adboost
    result= []

    for i in range(len(blockClassifier)):                    #对于每个块,这里求的是第一个示例每个块的检测概率（经验概率）
        P=ap.adaBoostClassify(blockClassifier[i],[x[i] for x in dataMat])#blockClassifier没问题，输入的dataMat是二维数组
        #print(P)
        result.append([i,P])

#**********************测试*********************
    '''
    for i in range(len(blockClassifier)):                    #看上下两次输出是否相同
        P=ap.adaBoostClassify(blockClassifier[i],[x[i] for x in dataMat])
        print(P)
    '''
#**********************结束*********************

    result = sorted(result, reverse=True, key=lambda x:x[1])
    #print(result)
    return result


def getData(inteIma, blockInfo, getRange,features):
    #inteIma 是积分图
    #blockInfo是上一帧目标位置第一块的信息
    #getRange是搜索范围信息
    #features是在每个块的哪个位置建立的特征值。这是个二维数组，第一维是每个随机蕨，第二位是随机蕨中每个特征值
    originlenth = blockInfo['lenth']
    originWidth = blockInfo['width']
    blocks = []
    tureFeats=[]

    dataMats =[]                                         #每个检测块的数据
    for magnification in range(-10,11):                   #对于块的浮动大小。长宽从缩小10%到放大10%

        lenth = int(originlenth * (1 + (magnification / 100)))  # 块的长计算
        width = int(originWidth * (1 + (magnification / 100)))  # 块的宽计算
        #print(originlenth,lenth)
        if magnification>-10:      #第一取样（第一次循环）时不判断
            if lenth == blocks[-1]['lenth'] and width == blocks[-1]['width']:
                continue    #前后两次取样长宽差别太小（相差不足1），跳过

        #print("长要搜索",len(range(getRange[0], getRange[1]-lenth+1)))
        x=getRange[0]
        while x<=getRange[1]-lenth+1:
        #for x in range(getRange[0], getRange[1]-lenth+1):  # 对于起点横坐标从开始到结束
            #print("宽要搜索",len(range(getRange[2], getRange[3]-width+1)))
            y=getRange[2]
            while y<=getRange[3]-width+1:
            #for y in range(getRange[2], getRange[3]-width+1): # 对于起始位置纵坐标，从开始到结束。块的起点位置确定完毕。
                dataMat=[]  # 一个块里建多个蕨。里面装的是data
                #print("搜索随机蕨的个数：",len(features))
                block ={}
                block['x'] = x
                block['y'] = y
                block['lenth'] = lenth
                block['width'] = width                #块的基本信息
                blocks.append(block)
                for featurelist in features:      #对于同一个块上不同的随机蕨.
                    # 从特征的相对位置到实际位置转换（不完全）
                    tureFeat=lt.getFeature(featurelist, block)
                    #计算特征值
                    #print(tureFeat)
                    #print()
                    data=[]  # 每个蕨里的特征数据，按顺序排列，顺序与特征位置数组相同
                    #print("搜索每个蕨特征的个数：",len(tureFeat))
                    for feature in tureFeat:               # 对于一个随机蕨的不同特征
                        value=it.towBitBP(
                            inteIma, block['x'] + feature['x'], block['y'] + feature['y'], feature['lenth'], feature['width'])
                        data.append(value)
                    dataMat.append(data)
                dataMats.append(dataMat)
                #此处dataMat第一维是每个块（此处应理解为示例），第二维是每个随机蕨，第三维是蕨内容
                y+=int(width*0.1)
            x+=int(lenth*0.1)
    #print(dataMats)
    return dataMats, blocks

def getDataTem(inteIma, features, blocksInfo):
    dataMats=[]
    for blockInfo in blocksInfo:   #对于每个块
        dataMat=[]
        for featurelist in features:     #下面这四行我自己看着都晕，慢慢琢磨吧。
            tureFeat=lt.getFeature(featurelist, blockInfo)
            data=[]
            for feature in tureFeat:
                value=it.towBitBP(
                    inteIma, blockInfo['x'] + feature['x'], blockInfo['y'] + feature['y'], feature['lenth'], feature['width'])
                data.append(value)
            dataMat.append(data)
        dataMats.append(dataArray)
    return dataMats

def getList(probability,maxP):    #计算maxP在probability中多次出现的位置，返回一个列表
    maxList=[]
    for i in range(len(probability)):
        if probability[i] == maxP:
            maxList.append(i)
    return maxList
