'''
    应用模块，此处为应用学习到的东西的代码模块。
'''
import numpy as np


def fernClassify(randomFern, dataMat, p=0):          #应用蕨分类器。p是样本数据在蕨中不存在，应该返回什么概率。默认为0。
    lables = []
    dataMat = np.array(dataMat)
    for data in dataMat:
        while(True):
            feat = randomFern.key()                  #这一个子蕨的根元素
            print(feat)
            value = data[feat]                       #数据
            tem = randomFern[feat].get(value,False）
            if(tem):                                 #如果这个数据在蕨中存在
                randomFern = tem                     #进入下一个子蕨
            else: lables.append(p)                  
            if(type(randomFern) != dict):            #如果下一个子蕨不是字典（而是个数，应该是数字吧）
                print(randomFern)
                lables.append(randomFern)
    return lables

    def adaBoostClassify(classifiers,dataMat):
        for classifier in classifiers:
            fernClassify(classifier["fern"], dataMat)
    pass
