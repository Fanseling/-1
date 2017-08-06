'''
    应用模块，此处为应用学习到的东西的代码模块。
'''
import numpy as np


def fernClassify(randomFern, dataMat):
    lables = []
    dataMat = np.array(dataMat)
    for data in dataMat:
        while(True):
            feat = randomFern.key()
            print(feat)
            value = data[feat]
            randomFern = randomFern[feat][value]
            if(type(randomFern) != dict):
                print(randomFern)
                return randomFern

    return lables
