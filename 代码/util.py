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

def choseBestBlock(blockClassifier,dataMat):
    P=adaBoostClassify(blockClassifier,dataMat)       #××××××××可能能是错的×××××××××仔细看看这个函数里dataMat到底是什么
