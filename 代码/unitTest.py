import numpy as np
from ImageTool import towBitBP  # 此处彻底破坏了之前的低耦合想法
import application as ap
import ImageTool as it
import learningTool as lt 
import util
import math


'''
mat = mat(zeros((8,6)))
for i in range(8):
    for j in range(6):
            mat[i,j] = random.randint(1,5)
print (mat)
#uprightMat,tiltMat=ImageTool.getInteIma(mat)
imageBlocks=[]
imageBlocks,blockInfo,offsetInfo=ImageTool.imageFrag(mat,0,0,4,8,2,4)
#print(blockInfo)
#print(offsetInfo)
#for i in range(len(imageBlocks)):
#    print(imageBlocks[i])
objectInf=ImageTool.objectConfirm(3,1,blockInfo[0:4],offsetInfo[0:4])
print(objectInf)
#print(uprightMat)

#print(ImageTool.towBitBP(uprightMat,0,0,4,4))

imageMat = mat(ImageTool.image2Mat('0001.jpg',1))
x=118
y=57
lenth=82
width=98
#x=x-lenth/2+1
#y=y-width/2+1

imageMat = ImageTool.mark(imageMat,x,y,lenth,width)
x=int(x-lenth*0.1)
y = int(y - width*0.1)
lenth = int(lenth*1.2)
width = int(width*1.2)
imageMat = ImageTool.mark(imageMat,x,y,lenth,width)
ImageTool.showIma(imageMat)
'''


# 确认目标最终位置，注意输入的要是跟踪块的块信息.x_t,y_t为上一帧的目标中心坐标，offsetInfo是上一帧的偏移信息
'''
def objectConfirm(x_t, y_t, blockInfo, offsetInfo):
    m = len(blockInfo)  # 以局部预测的整体位置
    objectInf = {}
    max_w = -inf
    min_w = inf
    objectX = 0
    objectY = 0
    object_len = 0
    object_wid = 0
    for i in range(m):
        object_tem = {}
        temX = blockInfo[i]['x'] + offsetInfo[i]['ox']
        temY = blockInfo[i]['y'] + offsetInfo[i]['oy']
        temlen = blockInfo[i]['lenth'] * offsetInfo[i]['ol']
        temwid = blockInfo[i]['width'] * offsetInfo[i]['ow']
        objectX += temX / 4  # 由子块预测目标的位置,采取平均数形式
        objectY += temY / 4
        object_len += temlen / m
        object_wid += temwid / m
    objectInf['x'] = int(objectX)
    objectInf['y'] = int(objectY)
    objectInf['lenth'] = int(object_len)
    objectInf['width'] = int(object_wid)
    return objectInf

blockInfo = []
tem={}
tem['x']=3
tem['y']=0
tem['lenth']=2
tem['width']=2
blockInfo.append(tem)
tem['x'] = 3
tem['y'] = 2
tem['lenth'] = 2
tem['width'] = 2
blockInfo.append(tem)
tem['x'] = 1
tem['y'] = 0
tem['lenth'] = 2
tem['width'] = 2
blockInfo.append(tem)
tem['x'] = 1
tem['y'] = 2
tem['lenth'] = 2
tem['width'] = 2
blockInfo.append(tem)

x_t=2
y_t=3

offsetInfo=[]
tem = {}
tem['ox'] = -1
tem['oy'] = 3
tem['ol'] = 2
tem['ow'] = 4
offsetInfo.append(tem)
tem['ox'] = -1
tem['oy'] = 1
tem['ol'] = 2
tem['ow'] = 4
offsetInfo.append(tem)
tem['ox'] = 1
tem['oy'] = 3
tem['ol'] = 2
tem['ow'] = 4
offsetInfo.append(tem)
tem['ox'] = 1
tem['oy'] = 1
tem['ol'] = 2
tem['ow'] = 4
offsetInfo.append(tem)
a=objectConfirm(x_t, y_t, blockInfo, offsetInfo)
print(a['x'])
print(a['y'])
print(a['lenth'])
print(a['width'])
'''
#getPosBag测试
'''
def getPosBag(x, y, lenth, width, proportion=0.2):  # 取得正包， x,y是最左上角的坐标
    a = int(x - lenth * proportion / 2)
    x_star = a
    x_end = int(x + lenth * proportion / 2)
    y_star = int(y - width * proportion / 2)
    y_end = int(y + width * proportion / 2)
    tem = {}
    positiveBag = []
    tem['x'] = x
    tem['y'] = y
    positiveBag.append(tem)  # 将初始图片放在第一个位置。下面是其他正示例。
    while y_star <= y_end:
        #print(y_star)
        while x_star <= x_end:
            #print(x_star)
            tem = {}
            tem['x'] = x_star
            tem['y'] = y_star
            #print(y_star)
            positiveBag.append(tem)
            #print(tem)
            x_star += 2 
        y_star += 2
        x_star = a
    return positiveBag


a = getPosBag(100, 100, 70, 70)
print(len(a))
print(a)
'''

#getNegBag测试
'''
def getNegBag(x, y, lenth, width, proportion=0.2):  # 取得负包，参数与上面一模一样
    x_star = int(x - lenth * proportion / 2) - lenth
    x_end = int(x + lenth * proportion / 2) + lenth
    y_star = int(y - width * proportion / 2) - width
    y_end = int(y + width * proportion / 2) + width
    negativeBag = []
    
    y = y_star
    while y <= y_end:
        tem = {}
        tem['x'] = x_star
        tem['y'] = y
        negativeBag.append(tem)
        tem = {}
        tem['x'] = x_end
        tem['y'] = y
        negativeBag.append(tem)
        y += int(width * 0.1)  # 固定每隔10%个像素，取一个负示例
    x - x_star
    while x <= x_end:
        tem = {}
        tem['x'] = x
        tem['y'] = y_star
        negativeBag.append(tem)
        tem = {}
        tem['x'] = x
        tem['y'] = y_end
        negativeBag.append(tem)
        x += int(lenth * 0.1)
    return negativeBag


a = getNegBag(100, 100, 100, 100)
print(len(a))
print(a)
'''

'''
def getPosition(path):  # (文件操作)从给定的txt中，获取目标位置，注意，path是图片所在文件夹，要返回上一级

    path = path + "groundtruth_rect.txt"
    tar = []  # 目标位置的列表，实在想不到单词了
    with open(path,'r') as tarfile:
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


a = getPosition("G:\\课题\\test\\")
print(a)
'''

'''
def randomSelect(lenth, width, num):  # 随机挑选y用于组成随机蕨的特征。参数为图像块的长宽，特征数量。在图像块上选取。
    randFeat = []
    for i in range(num):
        tem={}
        lastX = math.ceil(2 / lenth)  # 两个像素占长的百分之多少
        lastY = math.ceil(2 / width)
        # 由于图像从第0个像素开始，所以是0-100。计算具体像素点位置是长宽记得减1
        randX = np.random.randint(0, 100 - lastX)
        randY = np.random.randint(0, 100 - lastY)
        lastLen = 100 - randX
        lastWid = 100 - randY
        randLen = np.random.randint(lastX, lastLen)  # lastX....好好想想吧
        randWid = np.random.randint(lastY, lastWid)
        tem['x'] = randX
        tem['y'] = randY
        tem['lenth'] = randLen
        tem['width'] = randWid
        randFeat.append(tem)
    return randFeat


a=randomSelect(20, 40, 6)
print(a)
'''





imagePath = "G:\\课题\\test\\ima"
imaMat = it.image2Mat(imagePath + '\\' + "0001.jpg", 1)
inteIma = it.getInteIma(imaMat)
print(inteIma)
