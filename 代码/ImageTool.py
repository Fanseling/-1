'''
图像相关工具函数。此处的“已测试”是指单元测试，整体是否有逻辑漏洞不保证
'''
from numpy import *
import matplotlib.pyplot as plt
from PIL import Image

def strat2center(x,y,lenth,width):                         #将起点坐标xy，变为中心坐标xy,已测试
    x = x + int(lenth/2)
    y = y + int(width/2)
    if lenth % 2 == 0 : x = x-1
    if width % 2 == 0 : y = y-1
    return x,y

def center2start(x,y,lenth,width):
    if lenth % 2 == 0 : x = x+1
    if width % 2 == 0 : y = y+1
    x = x - int(lenth/2)
    y = y - int(width/2)
    return x,y

def getInteIma(image):                                  #计算积分图      已通过单元测试
    image=mat(image)
    m,n=shape(image)
    uprightMat = mat(zeros((m+1,n+1)))
    tiltMat = mat(zeros((m+1,n+2)))
    #print(uprightMat)
    for i in range(1,m+1):                              #竖直特征积分图计算,逐行计算,第一行与第一列为0
        for j in range(1,n+1):
            #print(i,j)
            uprightMat[i,j]=uprightMat[i-1,j]+uprightMat[i,j-1]-uprightMat[i-1,j-1]+image[i-1,j-1]

    '''
    for i in range(1,m+1):
        for j in range(2,n+2):                              #45度倾斜积分图计算，逐列计算,前两列和第一行是全0
            #print(i,j)
            if j == 2:
                tiltMat[i,j]=image[i-1,j-2]
            elif i == m:
                tiltMat[i,j] = tiltMat[i-1,j-1]+image[i-1,j-2]+image[i-1,j-3]
            else:
                tiltMat[i,j] = tiltMat[i-1,j-1]+tiltMat[i+1,j-1]-tiltMat[i,j-2]+image[i-1,j-2]+image[i-1,j-3]
    '''
    return uprightMat#,tiltMat

#def calcuFeature(x,y,featureType):                      #计算目标区域特征值...我擦，蕨分类不用haar-like！作废

def towBitBP(uprightMat,x,y,lenth,width):                 #X,Y是矩阵左上角的坐标     已通过单元测试,但是....应该没问题吧？
    upright = mat(uprightMat)
    w,l=shape(uprightMat)
    w=w-1
    l=l-1
    x=x+1                                                 #积分图的有效数据是从（1,1）开始，转换一下
    y=y+1
    x1=int(x+lenth/2)-1
    x2=int(x+lenth)-1
    y1=int(y+width/2)-1
    y2=int(y+width)-1
    if x1<0 : x1=0
    if x2<0 : x2=0
    if y1<0 : y1=0
    if y2<0 : y2=0
    if y-1<0 : y=1
    if x-1<0 : x=1
    if x1>l : x1=l
    if x2>l : x2=l
    if y1>w : y1=w
    if y2>w : y2=w
    if y>w : y=w
    if x>l : x=l
    len1 =upright[y2,x1]-upright[(y-1),x1]-upright[y2,(x-1)]+upright[(y-1),(x-1)]   #注意，矩阵先行后列，先y后x
    len2 =upright[y2,x2]-upright[(y-1),x2]-upright[y2,x1]+upright[(y-1),x1]
    if len1-len2<0 : symbol1 = '1'
    else : symbol1 = '0'
    wid1 = upright[y1,x2]-upright[(y-1),x2]-upright[y1,(x-1)]+upright[(y-1),(x-1)]
    wid2 = upright[y2,x2]-upright[y1,x2]-upright[y2,(x-1)]+upright[y1,(x-1)]
    if wid1-wid2<0 : symbol2 = '1'
    else : symbol2 = '0'
    symbol = symbol1+symbol2
    symbol = '0b'+symbol
    return int(symbol,2)

def imageFrag(x,y,lenth,width,xBlockNum,yBlockNum):           #图像分块,已测试
    #if lenth%xBlockNum!=0:lenth=lenth%xBlockNum      #保证目标区域可以偶数分块,这里是配合x，yBlockNum必然
    #if width%yBlockNum!=0:width+=width%yBlockNum     #是偶数使用的,保证除尽
    #imageBlocks=[]
    blockInfo = []                                   #块信息：中心坐标，长、宽
    offsetInfo =[]                                   #块的偏移信息：中心偏移，宽、高比值
         #这里的中心不是绝对中心，中心的左边比右边，上边比下边少一个像素
    m = int(lenth/xBlockNum)                     #m,n是子块的长和宽,m是长，n是宽
    n = int(width/yBlockNum)
    index = 0
    centerX,centerY = strat2center(x,y,lenth,width)
    for i in range(yBlockNum):
        for j in range(xBlockNum):
            #imageBlock = mat(zeros((n,m)))
            temX = x+j*m      #计算子块的开始坐标。已经核实很多遍了，这个是对的！
            temY = y+i*n
            #print(image[temY:temY+n,temX:temX+m])
            InfoTem={}
            offsetTem = {}
            InfoTem['x'],InfoTem['y'] = strat2center(temX,temY,m,n)                   #算子块的中心坐标
            InfoTem['lenth'] = m
            InfoTem['width'] = n
            offsetTem['ox'] = centerX - InfoTem['x']
            offsetTem['oy'] = centerY - InfoTem['y']
            InfoTem['x'] = temX
            InfoTem['y'] = temY
            offsetTem['ol'] = xBlockNum
            offsetTem['ow'] = yBlockNum
            #imageBlock[:,:]=image[temY:temY+n,temX:temX+m]
            #imageBlocks.append(imageBlock)
            blockInfo.append(InfoTem)
            offsetInfo.append(offsetTem)
    return blockInfo,offsetInfo#,imageBlocks                     返回的是块的位置信息，和偏倚信息，需要进一步处理。

def objectConfirm(x_t,y_t,blockInfo,offsetInfo):       #确认目标最终位置，注意输入的要是跟踪块的块信息.x_t,y_t为上一帧的目标中心坐标，offsetInfo是上一帧的偏移信息
    m= len(blockInfo)                                  #以局部预测的整体位置 已测试
    objectInf = {}
    max_w = -inf
    min_w = inf
    objectX = 0
    objectY = 0
    object_len = 0
    object_wid = 0
    for i in range(m):
        centerX,centerY = strat2center(blockInfo[i]['position']['x'],blockInfo[i]['position']['y'],blockInfo[i]['position']['lenth'],blockInfo[i]['position']['width'])
        object_tem = {}
        temX = centerX+offsetInfo[blockInfo[i]['blockIndex']]['ox']
        temY = centerY+offsetInfo[blockInfo[i]['blockIndex']]['oy']
        temlen = blockInfo[i]['position']['lenth']*offsetInfo[blockInfo[i]['blockIndex']]['ol']
        temwid = blockInfo[i]['position']['width']*offsetInfo[blockInfo[i]['blockIndex']]['ow']
        objectX += temX/m                        #由子块预测目标的位置,采取平均数形式
        objectY += temY/m
        object_len += temlen/m
        object_wid += temwid/m
    objectX,objectY=center2start(int(objectX),int(objectY),int(object_len),int(object_wid))
    objectInf['x'] = objectX
    objectInf['y'] = objectY
    objectInf['lenth'] = int(object_len)
    objectInf['width'] = int(object_wid)
        #print('x: %f, y: %f , lenth: %f, width: %f' %(temX,temY,temlen,temwid))
        #下面是按照论文写的，结果证明论文中的公式有问题
    '''
        temX = x_t-blockInfo[i]['x']                  #上一帧的中心坐标，减去这一帧、这一块预测的目标中心，用来计算w
        temY = y_t-blockInfo[i]['y']
        tem = temX**2+temY**2
        tem = -sqrt(tem)
        tem = exp(tem)
        #print(tem)
        object_tem['w'] = tem
        if tem>=max_w : max_w=tem
        if tem<min_w : min_w=tem
        object_k.append(object_tem)
    objectInf = {}
    objectInf['x'] = 0                                  #目标最终位置信息初始化
    objectInf['y'] = 0
    objectInf['lenth'] = 0
    objectInf['width'] = 0
    for i in range(m):
        object_k[i]['w'] = (object_k[i]['w']-min_w)/(max_w-min_w)
        print(object_k[i]['w'])
        print(object_k[i]['x'])
        objectInf['x'] += object_k[i]['w']*object_k[i]['x']
        objectInf['y'] += object_k[i]['w']*object_k[i]['y']
        objectInf['lenth'] += object_k[i]['w']*object_k[i]['lenth']
        objectInf['width'] += object_k[i]['w']*object_k[i]['width']
    halfLen = objectInf['lenth']/2
    halfWid = objectInf['width']/2

    if objectInf['lenth']%2 == 0 : halfLen = halfLen-1     #直径是偶数，那中心点其实在真正中心的左上角一格，画个矩阵就明白了
    if objectInf['width']%2 == 0 : halfWid = halfWid-1
    objectInf['x'] = objectInf['x']-halfLen               #将xy的坐标变为最左上角的坐标，即start处
    objectInf['y'] = objectInf['y']-halfWid
    '''
    return objectInf

def getPosBag(x,y,lenth,width,proportion=0.2):                #取得正包， x,y是最左上角的坐标

    x_star = int(x - lenth * proportion / 2)
    x_end = int(x + lenth * proportion/2)
    y_star = int(y - width * proportion/2)
    y_end = int(y + width * proportion/2)
    tem = {}
    positiveBag = []
    tem['x'] = x
    tem['y'] = y
    tem['lenth'] = lenth
    tem['width'] = width
    positiveBag.append(tem)                                 #将初始图片放在第一个位置。下面是其他正示例。
    while y_star <= y_end:
        while x_star <= x_end:
            tem={}
            tem['x'] = x_star
            tem['y'] = y_star
            tem['lenth'] = lenth
            tem['width'] = width
            positiveBag.append(tem)
            x_star += 1                #×××××××××用固定像素可能不太好×××××××××××××
        y_star += 1
        x_star = a
    return positiveBag                                       #结构：列表，每个列表元素是字典，是正包起始位置xy

def getNegBag(x,y,lenth,width,xBlockNum,yBlockNum,proportion=0.2):                   #取得负包，
    blockLen = int(lenth/xBlockNum)
    blockWid = int(width /yBlockNum)
    x_star = int(x - lenth * proportion/2) - blockLen
    x_end = int(x + lenth * proportion/2) + lenth
    y_star = int(y - width * proportion/2) - blockWid
    y_end = int(y + width * proportion/2) + width
    negativeBag = []
    y = y_star
    while y <=y_end:
        tem = {}
        tem['x'] = x_star
        tem['y'] = y
        tem['lenth'] = blockLen
        tem['width'] = blockWid
        negativeBag.append(tem)
        tem = {}
        tem['x'] = x_end
        tem['y'] = y
        tem['lenth'] = blockLen
        tem['width'] = blockWid
        negativeBag.append(tem)
        y += int(blockWid*0.1)                                  #固定每隔20%个像素，取一个负示例
    x - x_star
    while x <=x_end:
        tem = {}
        tem['x'] = x
        tem['y'] = y_star
        tem['lenth'] = blockLen
        tem['width'] = blockWid
        negativeBag.append(tem)
        tem = {}
        tem['x'] = x
        tem['y'] = y_end
        tem['lenth'] = blockLen
        tem['width'] = blockWid
        negativeBag.append(tem)
        x += int(blockLen*0.1)
    return negativeBag

def image2Mat(path,color):                                   #图片转矩阵,已测试
    im=Image.open(path)
    if color == 3:im = im.convert('L')
    lenth,width = im.size
    imagedata = im.getdata()
    imagedata = mat(imagedata)
    imagedata = reshape(imagedata,(width,lenth))
    return imagedata

def showIma(imaMat):                                        #把矩阵作为图片显示（黑白）,已测试
    image = Image.fromarray(imaMat.astype(uint8))
    image.show()

def mark(imaMat,imaDict):#在图片上标记以xy为起点，lenth,width的矩形区域，x,y,lenth,width存在一个字典传进函数。已测试
    #imaMat=imaMatTem.copy()
    x=imaDict['x']
    y=imaDict['y']
    lenth = imaDict['lenth']
    width = imaDict['width']
    imaMat = mat(imaMat)
    m,n=shape(imaMat)
    m=m-1
    n=n-1
    if x<0:x=0
    if y<0:y=0
    x=min(x,n)
    y=min(y,m)
    for i in range(lenth):
        imaMat[y,min(x+i,n)] = 0
        imaMat[min(y+width-1,m),min(x+i,n)] = 0
    for j in range(width):
        imaMat[min(y+j,m),x] = 0
        imaMat[min(y+j,m),min(x+lenth-1,n)] =0
    return imaMat
