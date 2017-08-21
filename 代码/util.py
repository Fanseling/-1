'''
通用工具模块
'''
import application as ap

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

def blockSortedByP(blockClassifier,dataMat):          #已测试，根据每个块在学习到的分类器上的应用，得出概率，根据概率排序
    P=adaBoostClassify(blockClassifier,dataMat)       #输入的dataMat是二维数组
    result= []
    for i in range(len(P)):
        result.append([i,P[i]])
    result=sorted(result, reverse=True key=lambda x:x[1])
    return result
