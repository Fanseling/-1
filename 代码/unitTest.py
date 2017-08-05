from numpy import *
import ImageTool
from PIL import Image
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
'''
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
