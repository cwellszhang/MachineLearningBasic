# -*- coding: utf-8 -*-
from numpy import *
class regTree:
    """
    这里只实现了回归树。分类树的实现与id3_c45类似，只是改用基尼系数来估计。
    """
    def __init__(self):
        self.regTree = None

    def loadDataSet(self,fileName):
        dataMat = []
        fr = open(fileName)
        for line in fr.readlines():
            curLine = line.strip().split("\t")
            fltLine = map(float,curLine) #map all elements to float() dataMat.append(fltLine)
            dataMat.append(fltLine)
        return dataMat

    def _regLeaf(dataSet):
        """
        该函数负责生成叶节点。在回归树中，该模型其实就是目标变量的均值。
        """
        return mean(dataSet[:,-1])

    def _regErr(dataSet):
        """
        该函数在给定数据集上计算目标变量的平方误差。
        这里直接调用均方差函数var(),因为需要返回的是总方差，所以要用均方差乘以数据集中样本的个数。
        """
        return var(dataSet[:,-1]) * shape(dataSet)[0]

    def _chooseBestSplit(self,dataSet,leafType=_regLeaf,errType=_regErr,ops=(1,4)):
        """
        该函数的目的是找到数据的最佳二元切分方式。
        如果找不到一个好的二元切分，该函数返回None,并同时调用createTree()方法来产生叶节点，叶节点的值也将返回None。
        在函数中有三种情况不会切分，而是直接创建叶节点。
        如果找到了好的切分方式，则返回特征编号和切分特征值
        :parameter tolS :容许的误差下降值
        :parameter tolN :切分的最少样本数
        """
        tolS = ops[0];tolN = ops[1]
        if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
            return None,leafType(dataSet)
        m,n = shape(dataSet)
        S = errType(dataSet)
        bestS = inf;bestIndex = 0; bestValue=0
        for featIndex in range(n-1):
            # print( set(array(dataSet[:,featIndex])))
            for splitVal in set( dataSet[:,featIndex].T.A.tolist()[0]  )  :
                mat0,mat1 = self._binSplitDataSet(dataSet,featIndex,splitVal)
                if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN :
                    continue
                newS = errType(mat0) + errType(mat1)
                if newS < bestS :
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        #如果误差减少不大则退出
        if (S - bestS) < tolS:
            return  None,leafType(dataSet)
        mat0,mat1 = self._binSplitDataSet(dataSet,bestIndex,bestValue)
        #如果切分出的数据集很少则退出
        if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
            return None,leafType(dataSet)
        return bestIndex,bestValue


    def _binSplitDataSet(self,dataSet,feature,value):
        """
        :param dataSet: 数据集合
        :param feature: 待切分的特征
        :param value:   该特征的某个值
        :return: 该函数通过数组过滤的方式将上述数据集合切分得到两个子集并返回
        """
        mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
        mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
        return mat0,mat1

    def _createTree(self,dataSet,leafType = _regLeaf, errType=_regErr,ops=(1,4)):
        """
        :param dataSet: 数据集合
        :param leafType: 建立叶节点的函数
        :param errType:  误差计算函数
        :param ops: 函数树构建所需其他参数的元组

        如果构建的是回归树,该模型是一个常数。如果是模型树，其模型是一个线性方程。
        """
        feat,val = self._chooseBestSplit(dataSet,leafType,errType,ops)
        if feat == None:return val
        retTree = {}
        retTree['spInd'] = feat
        retTree['spVal'] = val
        lSet,rSet = self._binSplitDataSet(dataSet,feat,val)
        retTree['left'] = self._createTree(lSet,leafType,errType,ops)
        retTree['right'] = self._createTree(rSet,leafType,errType,ops)
        return retTree


    def _linearSolve(self,dataSet):
        """
        主要功能是将数据集格式化成目标变量Y和自变量X。
        """
        m,n = shape(dataSet)
        X = mat(ones((m,n)))
        Y = mat(ones((m,1)))
        X[:,1:n] = dataSet[:,0:n-1]
        Y = dataSet[:,-1]
        xTx = X.T*X
        if linalg.det(xTx) ==0.0:
            raise NameError('This matrix is a singular,connot do inverse\n')
        ws = xTx.I * (X.T*Y)
        return ws,X,Y
    def modelLeaf(self,dataSet):
        """
        负责生成叶结点，返回回归系数ws
        """
        ws,X,Y = self._linearSolve(dataSet)
        return ws
    def modelErr(self,dataSet):
        ws,X,Y = self._linearSolve(dataSet)
        yHat = X*ws
        return sum(power(Y-yHat,2))

    """
    树的剪枝
    """
    def isTree(self,obj):
        return type(obj).__name__=='dict'

    def getMean(self,tree):
        """
        递归函数，从上往下遍历树直到叶节点为止。如果找到两个叶节点则计算它们的平均值。
        该函数对树进行塌陷处理（即返回树的平均值）,在 prune函数中调用该函数时应明确。
        """
        if self.isTree(tree['right']):
            tree['right'] = self.getMean(tree['right'])
        if self.isTree(tree['left']):
            tree['left']=self.getMean(tree['left'])
        return (tree['right']+tree['right'])/2.0

    def prune(self,tree,testData):
        """
        prune首先确认测试集是否为空，一旦非空，则反复递归调用函数prune对测试数据进行切分。
        因为树是由其他数据集生成的，所以测试集上会有一些样本与原数据集样本的取值返回不同，这里假设发生了
        过拟合，对树进行剪枝。
        接下来要检查某个分支到底是子树还是节点。如果是子树，就调用函数prune来剪枝。对左右两个分支完成剪枝后，
        还要检查它们是否仍然还是子树。如果两个分支已经不再是子树，那么就可以进行合并。

        剪枝的效果并不一定比没有剪枝好。
        """
        if shape(testData)[0]==0 : return self.getMean(tree) #如果没有测试数据则对树进行塌陷处理
        if self.isTree(tree['right']) or self.isTree(tree['left']):
            lSet,rSet = self._binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        if self.isTree(tree['left']):
               tree['left'] = self.prune(tree['left'],lSet)
        if self.isTree(tree['right']):
               tree['right'] = self.prune(tree['right'],rSet)
        if not self.isTree(tree['left']) and not self.isTree(tree['right']):
            lSet,rSet = self._binSplitDataSet(testData,tree['spInd'],tree['spVal'])
            errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2))+\
                sum(power(rSet[:,-1]-tree['right'],2))
            treemean = (tree['left']+tree['right'])/2.0
            errorMerge = sum( power(testData[:,-1]-treemean,2))
            if errorMerge < errorNoMerge:
                print("merging")
                return treemean
            else:
                return tree
        else:
            return tree





if __name__ =='__main__':
    regTree = regTree()
    mymat = regTree.loadDataSet('data/ex2.txt')
    tree = regTree._createTree(mat(mymat),ops=(0,1))
    print(tree)

    mymat2 = regTree.loadDataSet('data/ex2test.txt')
    mymat2test = mat(mymat2)
    tree_pruned= regTree.prune(tree,mymat2test)
    print(tree_pruned)


    mymat = regTree.loadDataSet('data/exp2.txt')
    tree = regTree._createTree(mat(mymat),regTree.modelLeaf,regTree.modelErr,ops=(1,10))
    print(tree)
    """
    {'spInd': 0, 'spVal': 0.285477, 'right': matrix([[ 3.46877936],
        [ 1.18521743]]), 'left': matrix([[  1.69855694e-03],
        [  1.19647739e+01]])}

    """
