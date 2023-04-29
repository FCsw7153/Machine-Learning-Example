import numpy as np
import pandas as pd
import plottree

data = pd.DataFrame(pd.read_csv("watermelon_3_1.csv", encoding="utf-8"))
#print(data)
data.drop(labels = ["编号"], axis = 1, inplace = True)
data["好瓜"].replace(to_replace=["是", "否"], value=["好瓜", "坏瓜"], inplace=True) 
#print(data)

trainIndex = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16, 3]
traindata = data.loc[trainIndex]
#print(traindata)
testIndex = [4, 7, 8, 10, 11, 12]
testdata = data.loc[testIndex]


featurelist = traindata.columns[:-1]
#print(featurelsit)
featruevalue = {}
for f in featurelist:
    featruevalue[f]  = set(traindata[f])
#print(featurelist)
#print(featruevalue)
testfeaturelist = testdata.columns[:-1]
testfeatruevalue = {}
for f in testfeaturelist:
    testfeatruevalue[f]  = set(testdata[f])

def Gini(data):
    numEntries = data.shape[0]
    classArr = data.iloc[:, -1]
    uniqueClass = list(set(classArr))
    #print(uniqueClass)
    Gini = 1.0
    for c in uniqueClass:
        Gini -= (len(data[data.iloc[:, -1] == c]) / float(numEntries)) ** 2
        #print(len(data[data.iloc[:, -1] == c]))
    return Gini

def split(traindata, feature):
    splitD = []
    for df in traindata.groupby(by = feature, axis = 0):
        splitD.append(df)
    #print(splitD)
    return splitD 

def GiniIndex(traindata, feature):
    GiniIndex = 0.0
    return GiniIndex + sum(len(DV[1]) / len(traindata) * Gini(DV[1]) for DV in split(traindata, feature))


def choosebest(D, A):
    informationGain = {}
    for feature in A:
        informationGain[feature] = GiniIndex(D, feature)
    #print(informationGain)
    informationGain = sorted(informationGain.items(), key=lambda ig:ig[1], reverse=False)
    return informationGain[0][0]

def countMajority(D):
    return D["好瓜"].mode().iloc[0]

def treeGenerate(D, A):
    if len(split(D, "好瓜")) == 1: 
        return D["好瓜"].iloc[0]
    if len(A) == 0 or len(split(D, A.tolist())) == 1:
        return countMajority(D)
    bestFeature = choosebest(D, A)
    myTree = {bestFeature:{}}
    best = pd.unique(D[bestFeature])
    #print(best)
    if len(split(D, bestFeature)) != len(featruevalue[bestFeature]):
        no_exist_feature = set(featruevalue[bestFeature]) - set(best)
        for no_feature in no_exist_feature:
            myTree[bestFeature][no_feature] = countMajority(D)
    for bestFeatureValue, Dv in split(D, bestFeature):
        if len(Dv) == 0:
            return countMajority(D)
        else:
            A1 = pd.Index(A)
            A1 = A1.drop([bestFeature])
            Dv = Dv.drop(labels=[bestFeature], axis=1)
            myTree[bestFeature][bestFeatureValue] = treeGenerate(Dv, A1)
            
    return myTree

if __name__ == "__main__":
    myTree = treeGenerate(traindata, featurelist)
    #print(myTree)
    #plottree.createPlot(myTree)