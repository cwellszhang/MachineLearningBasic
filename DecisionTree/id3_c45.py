# -*- coding: utf-8 -*-
import numpy as np

class DecisionTree:
    """

    """
    def __init__(self,mode='C4.5'):
        self._tree = None

        if mode == 'C4.5' or mode == 'ID3':
            self._mode = mode
        else:
            raise Exception('mode should be C4.5 or ID3')


    def _calcEntropy(self,y):
        """
        :param y: 数据集的标签
        :return: 数据集的返回熵
        """
        numEntried = y.shape[0]
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys():
                labelCounts[label] = 0
            labelCounts[label] += 1
        # 计算熵
        entropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]/numEntried)
            entropy -= prob*np.log2(prob)
        return entropy

    def _splitDataSet(self,X,y,index,value):
        """
        :param X: 数据集的特征
        :param y: 数据集的标签
        :param index: 特征下标
        :param value: 按照value进行划分
        :return: 数据集中特征下标为axis,特征值等于value的子数据集
        """
        retDataSet  = []
        featVec = X[:,index]
        X = X[:,[i for i in range(X.shape[1]) if i!= index ]]
        for i in range(len(featVec)):
            if featVec[i]==value:
                retDataSet.append(i)
        return X[retDataSet,:],y[retDataSet]

    def _chooseBestFeatureToSplit_ID3(self,X,y):
        """
        输入数据集，选择最佳分割特征
        :param X: 数据集的特征
        :param y: 数据集的标签
        :return: 返回最佳分割的特征下标️
        """
        numFeatures = X.shape[1]
        baseEntropy  = self._calcEntropy(y)
        bestInfoGain = 0.0
        bestFeatureIndex= -1
        #对每一个特征计算信息增益，记录能使信息增益最大的特征
        for i in range(numFeatures):
            featList = X[:,i]
            uniqueVals = set(featList)
            newEntropy = 0.0
            #对第 i 个特征的每个value，划分子数据集并计算熵,得到信息增益
            for value in uniqueVals:
                sub_X,sub_y = self._splitDataSet(X,y,i,value)
                prob = len(sub_y)/float(len(y))
                newEntropy += prob*self._calcEntropy(sub_y)
            #计算信息增益
            infoGain = baseEntropy - newEntropy
            if infoGain>bestInfoGain:
                bestFeatureIndex=i
                bestInfoGain = infoGain
        return bestFeatureIndex


    def _chooseBestFeatureToSplit_C45(self,X,y):
        """
        ID3采用信息增益，C4.5采用的是信息增益比
        """
        numFeatures = X.shape[1]
        baseEntropy  = self._calcEntropy(y)
        bestGainRatio = 0.0
        bestFeatureIndex= -1
        #对每一个特征计算信息增益比，gainRatio=infoGain/splitInfomation
        for i in range(numFeatures):
            featList = X[:,i]
            uniqueVals = set(featList)
            newEntropy = 0.0
            splitInformation = 0.0
            #对第 i 个特征的每个value，划分子数据集并计算熵,得到信息增益
            for value in uniqueVals:
                sub_X,sub_y = self._splitDataSet(X,y,i,value)
                prob = len(sub_y)/float(len(y))
                newEntropy += prob*self._calcEntropy(sub_y)
                splitInformation -= prob*np.log2(prob)
            #计算信息增益比
            #若splitInfomation为0，说明该特征的所有值都是相同的，不能做分割特征
            if splitInformation==0.0:
                pass
            else:
                infoGain = baseEntropy - newEntropy
                gainRatio = infoGain/splitInformation
                if gainRatio>bestGainRatio:
                    bestFeatureIndex=i
                    bestGainRatio = gainRatio
        return bestFeatureIndex

    def _majorityCnt(self,labelList):
        """
        :param labelList: 标签的列表
        :return: 标签中出现次数最多的label
        """
        labelCount={}
        for vote in labelList:
            if vote not in labelCount.keys():
                labelCount[vote]=0
            labelCount[vote]+=1
        sortedLabelCount = sorted(labelCount.iteritems(),key=lambda x:x[1],reverse=True)
        return sortedLabelCount[0][0]

    def _createTree(self,X,y,featureIndex):
        """
        :param X:  数据集的特征
        :param y:  数据集的标签
        :param featureIndex: 特征列表的下表
        :return: 决策树
        """

        labelList=list(y)
        #如果所有label都相同,则停止划分,返回该label
        if labelList.count(labelList[0]) == len(labelList):
            return labelList[0]
        #如果只有一个特征，则返回出现次数最多的label
        if len(featureIndex)==1:
            return self._majorityCnt(labelList)
        #可以继续划分的话，确定最佳分割特征
        if self._mode =='C4.5':
            bestFeatIndex = self._chooseBestFeatureToSplit_C45(X,y)
        elif self._mode == 'ID3':
            bestFeatIndex = self._chooseBestFeatureToSplit_ID3(X,y)

        bestFeatStr = featureIndex[bestFeatIndex]
        featureIndex = list(featureIndex)
        featureIndex.remove(bestFeatStr)
        featureIndex = tuple(featureIndex)
        #用字典存储决策树，最佳分割特征作为key,而对应的键值仍然是一棵树
        myTree = {bestFeatStr:{}}
        featValues = X[:,bestFeatIndex]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            #对每个value递归地构建一棵树
            sub_X,sub_y = self._splitDataSet(X,y,bestFeatIndex,value)
            myTree[bestFeatStr][value]=self._createTree(sub_X,sub_y,featureIndex)
        return myTree

    def fit(self,X,y):
        #类型检查
        if isinstance(X,np.ndarray) and isinstance(y,np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
                y = np.array(y)
            except:
                raise TypeError("numpy.ndarray required for X,y")
        featureIndex = tuple(['x'+str(i) for i in range(X.shape[1])])
        self._tree = self._createTree(X,y,featureIndex)
        return self

    def predict(self,X):
        if self._tree ==None:
            raise NotFittedError("Estimator not fitted,call 'fit first")

        if isinstance(X,np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        def _classify(tree,sample):
            """
            决策树的构建是一个递归的过程，用决策树分类也是一个递归的过程
            _classify()一次只能对一个样本分类
            """
            featIndex = list(tree.keys())[0]
            secondDict = tree[featIndex]
            key = sample[int(featIndex[1:])]
            valueOfkey = secondDict[key]
            if isinstance(valueOfkey,dict):
                label = _classify(valueOfkey,sample)
            else:
                label = valueOfkey
            return label

        if len(X.shape)==1:
            return _classify(self._tree,X)
        else:
            results=[]
            for i in range(X.shape[0]):
                results.append(_classify(self._tree,X[i]))
            return np.array(results)


class NotFittedError(Exception):
    """
    Exception class to raise if estimator is used before fitting

    """
    pass

if __name__ == '__main__':
    X = [[1,1,'yes'],
       [1,1,'yes'],
       [1,0,'no'],
       [0,1,'no'],
       [0,1,'no']]
    y = ['no surfacing','no surfacing','flippers','flippers','flippers']
    tree = DecisionTree(mode='ID3')
    tree.fit(X,y)
    print(tree._tree)
    X=[1,0,'no']
    print(tree.predict(X))
