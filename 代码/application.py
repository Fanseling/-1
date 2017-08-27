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
            if(type(randomFern) != dict):            #如果下一个子蕨不是字典（而是个float数，应该是数字吧）
                print(randomFern)
                lables.append(randomFern)
    return lables                                     #返回的是一个随机蕨的全部分类结果，是个向量

    def adaBoostClassify(classifiers, dataMat) :     #应用强分类器（boosting）
        probability= np.array(np.zeros(len(dataMat)))
        weight = 0
        for classifier in classifiers:              #对于adaboost中的每个随机蕨
        #此处要找到每个随机蕨对应的dataMat
            probability += np.array(fernClassify(classifier["fern"], [data[classifier["index"]] for data in dataMat])) * classifier["alpha"] 
            weight+=classifier["alpha"]
        probability=probability/weight
        return probability
