import numpy as np
import pandas as pd
import plottree

data = pd.DataFrame(pd.read_csv("watermelon_3.csv", encoding="utf-8"))
#print(data)
data.drop(labels = ["编号"], axis = 1, inplace = True)
data["好瓜"].replace(to_replace=["是", "否"], value=["好瓜", "坏瓜"], inplace=True) 
#print(data)

'''
对于数据预处理，提取出所有的属性
'''
featurelist = data.columns[:-1]
#print(featurelsit)
featruevalue = {}
for f in featurelist:
    featruevalue[f]  = set(data[f])

'''
对于连续变量进行排序以及重新设置索引
'''
T = {} 
for feature in featurelist[-2:]:  
    T1 = data[feature].sort_values()
    T2 = T1.iloc[:-1].reset_index(drop=True) 
    T3 = T1.iloc[1:].reset_index(drop=True)  
    T[feature] = (T2+T3)/2

'''
计算属性的熵
'''
def ent(D):
    frequency = D["好瓜"].value_counts() / len(D["好瓜"])
    #print(frequency)
    return -sum(pk * np.log2(pk) for pk in frequency)

'''
分离出离散属性中的子集
'''
def split_discrete(D, feature):
    splitD = []
    for df in D.groupby(by = feature, axis = 0):
        splitD.append(df)
    return splitD 

'''
分离出连续属性
'''
def split_continue(D, feature, splitValue):
    splitD = []
    splitD.append(D[D[feature] <= splitValue])
    splitD.append(D[D[feature] > splitValue])
    return splitD

'''
计算离散属性的信息增益
'''
def gain_discrete(D, feature):
    gain = ent(D) - sum(len(DV[1]) / len(D) * ent(DV[1]) for DV in split_discrete(D, feature))
    return gain

'''
计算连续属性的信息增益
'''
def gain_continue(D, feature):
    _max = 0
    splitValue = 0
    for t in T[feature].values:
        temp = ent(D) - sum(len(DV) / len(D) * ent(DV) for DV in split_continue(D, feature, t))
        if _max < temp:
            _max = temp
            splitValue = t
    return _max, splitValue

'''
选择信息增益最优的属性
'''
def choosebest(D, A):
    informationGain = {}
    for feature in A:
        if feature in ["密度", "含糖率"]: 
            ig, splitValue = gain_continue(D, feature)
            informationGain[feature+"<=%.3f"%splitValue] = ig
        else:
            informationGain[feature] = gain_discrete(D, feature)
    #print(informationGain)
    informationGain = sorted(informationGain.items(), key=lambda ig:ig[1], reverse=True)
    return informationGain[0][0]

def countMajority(D):
    return D["好瓜"].mode().iloc[0]
'''
决策树算法主函数
'''
def treeGenerate(D, A):
    if len(split_discrete(D, "好瓜")) == 1: 
        return D["好瓜"].iloc[0]
    if len(A) == 0 or len(split_discrete(D, A.tolist())) == 1:
        return countMajority(D)
    bestFeature = choosebest(D, A)
    if "<=" in bestFeature:
        bestFeature, splitValue = bestFeature.split("<=")
        myTree = {bestFeature+"<="+splitValue:{}}
        [D0, D1] = split_continue(D, bestFeature, float(splitValue))
        A0 = pd.Index(A)
        A1 = pd.Index(A)
        myTree[bestFeature+"<="+splitValue]["yes"] = treeGenerate(D0, A0)
        myTree[bestFeature+"<="+splitValue]["no"] = treeGenerate(D1, A1)
    else:  
        myTree = {bestFeature:{}}
        for bestFeatureValue, Dv in split_discrete(D, bestFeature):
            #print(bestFeatureValue)
            #print(Dv)
            if len(Dv) == 0:
                return countMajority(D)
            else:
                A2 = pd.Index(A)
                A2 = A2.drop([bestFeature])
                Dv = Dv.drop(labels=[bestFeature], axis=1)
                myTree[bestFeature][bestFeatureValue] = treeGenerate(Dv, A2)
    return myTree

if __name__ == "__main__":
    myTree = treeGenerate(data, featurelist)
    #print(myTree)
    plottree.createPlot(myTree)