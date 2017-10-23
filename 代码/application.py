'''
    应用模块，此处为应用学习到的东西的代码模块。
'''
import numpy as np


def fernClassify(randomFern, dataMat, p=0):          #应用蕨分类器。p是样本数据在蕨中不存在，应该返回什么概率。默认为0。
#dataMat第一维是示例，第二维是特征序列,毕竟这个函数是要执行分类，来判断一个示例群的标签
#randomFern是一个随机蕨
    lables = []
    #print(dataMat)
    for data in dataMat:                #data就是个特征序列了
        #print(data)
        randomFernTem = randomFern.copy()
        while(True):
            #print(randomFernTem)
            #print(data)
            feat = list(randomFernTem.keys())
            feat=feat[0]                             #这一个子蕨的根元素
            #print("feat是：",feat)
            value = data[feat]                       #数据
            #print("value是：",value)
            #print("子节点是：",randomFernTem[feat].get(value,False))
            tem = randomFernTem[feat].get(value,False)
            #print("运行到哪了？")
            if(tem or tem==0.0):                        #如果这个数据在蕨中存在
                randomFernTem = tem                     #进入下一个子蕨
            else:
                lables.append(p)
                break
            #print(randomFernTem)
            if(type(randomFernTem) != dict):            #如果下一个子蕨不是字典（而是个float数，应该是数字吧）
                #print("randomFernTem的一个叶子是：",randomFernTem)
                lables.append(randomFernTem)
                break
    return lables                                     #返回的是一个随机蕨的全部分类结果，是个向量

'''
dataMat第一维是蕨，第二维是每个蕨的特征值序列。这是第0个示例的，某个块的dataMat
Classifiers 是一个块的adaboost弱分类序列，即一个adaboost分类器
'''
def adaBoostClassify(classifiers, dataMat) :     #应用强分类器（boosting）
#dataMat第一维是蕨，第二维是每个蕨的特征值序列。这是第0个示例的，某个块的dataMat
#Classifiers 是一个块的adaboost弱分类序列,即一个adaboost分类器
    probability= 0
    weight = 0
    for classifier in classifiers :              #对于adaboost中的每个随机蕨
    #此处要找到每个随机蕨对应的dataMat
        #print("classifier是：",classifier)
        #print(type(classifier["index"]))

        probability += fernClassify(classifier["fern"], [dataMat[classifier["index"]]])[0] * classifier["alpha"]
        weight+=classifier["alpha"]
    probability=probability/weight
    #print(probability)
    return probability
