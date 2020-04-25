'''
Author : liwenyi
file : decision_tree
优点：计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关的数据特征
缺点：可能产生过度匹配问题。
适用数据类型：数值型和标称型
'''

'''
决策树一般流程
1、收集数据：可以使用任何方法
2、准备数据：树构造只适用于标称型数据，因此数值型数据必须离散化
3、分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期
4、训练算法：构造树的数据结构
5、测试算法：使用经验树计算错误率
6、使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好的理解数据的内在含义
'''


from math import log
class Tree:
    def __init__(self):
        pass

    # 计算给定数据集的香农熵
    def calShannonEnt(self,dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * log(prob,2)
        return shannonEnt

    # 创建数据集
    def createDataSet(self):
        dataSet = [
            [1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']
        ]
        labels = ['no surfacing','flippers']
        return dataSet,labels

    # 按照给定的特征划分数据集
    def splitDataSet(self,dataSet,axis,value):
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    # 选择最好的数据集划分方式
    def chooseBestFeaturesToSplit(self,dataSet):
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calShannonEnt(dataSet)
        bestInfoGain = 0.0
        bestFeatures = -1
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntroy = 0.0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet,i,value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntroy += prob * self.calShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntroy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeatures = i
        return bestFeatures



if __name__ == '__main__':
    dtree = Tree()
    myDat,labels = dtree.createDataSet()
    print(dtree.splitDataSet(myDat,0,0))
    print(dtree.chooseBestFeaturesToSplit(myDat))
    # print(myDat)
    # print(Tree().calShannonEnt(myDat))