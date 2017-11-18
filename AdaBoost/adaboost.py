# -*- coding: utf-8 -*-
from numpy import *

class Adaboost:

    def __init__(self):
        self._adaboost = None

    def _stumpClassify(self,dataMatrix,index,threshVal,threshIneq):
        """用于测试是否有某个值小于或大于我们的阈值
        :param dataMatrix:  输入数据集,matrix格式,下同
        :param index:   特征下标
        :param threshVal: 阈值
        :param threshIneq: 判断类型,lt or gt
        """
        retArray = ones((dataMatrix.shape[0],1))
        if threshIneq == 'lt':
            retArray[dataMatrix[:,index] <= threshVal] = -1.0
        else:
            retArray[dataMatrix[:,index] > threshVal] = -1.0
        return retArray

    def _buildStump(self,dataArr,classLabels,D):
        """
        :param dataArr: 数据集特征部分
        :param classLabels: 数据集标签
        :param D: 数据集权重
        :return: 最佳单层决策树
        """
        dataMatrix = mat(dataArr)
        labelMat =mat(classLabels).T
        m,n = dataMatrix.shape
        numSteps = 10.0;bestStump={};bestClasEst = mat(zeros((m,1)))
        minError  = inf
        for i in range(n):
            rangeMin = dataMatrix[:,i].min()
            rangeMax = dataMatrix[:,i].max()
            stepSize = (rangeMax-rangeMin)/numSteps
            for j in range(-1,int(numSteps)+1):
                for inequal in ['lt','gt']:
                    threshVal = (rangeMin+float(j)*stepSize )
                    #阈值一边的会被分类到-1，另一边的会被分类到+1.
                    predictVals = self._stumpClassify(dataMatrix,i,threshVal,inequal)
                    errArr = mat(ones((m,1)))
                    errArr[predictVals==labelMat]=0
                    weightedError  = D.T*errArr
    #                 print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error: %.3f" \
    #                       %(i, threshVal, inequal, weightedError)  )
                    if weightedError < minError:
                        minError = weightedError
                        bestClasEst = predictVals.copy()
                        bestStump['dim']=i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        return bestStump,minError,bestClasEst

    def fit(self,X,y,num_iters = 40):
        weakClassArr = []
        m = shape(X)[0]
        D = mat(ones((m,1))/m)
        aggClassEst  = mat(zeros((m,1)))
        for i in range(num_iters):
            bestStump,error,classEst= self._buildStump(X,y,D)
            print("D:",D.T)
            alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
            bestStump['alpha']=alpha
            weakClassArr.append(bestStump)
            print("classEst:",classEst.T)
            expon = multiply(-1*alpha*mat(y).T,classEst)
            D = multiply(D,exp(expon))
            D = D/D.sum()
            aggClassEst += alpha*classEst
            print("aggClassEst:",aggClassEst.T)
            aggErrors = multiply(sign(aggClassEst)!=mat(y).T,ones((m,1)))
            errorRate = aggErrors.sum()/m
            print("total error:",errorRate)
            if errorRate==0.0:break
        self._adaboost=weakClassArr
        return self

    def predict(self,X):
        if self._adaboost ==None:
            raise NotFittedError("Estimator not fitted,call 'fit first")

        if isinstance(X,ndarray):
            pass
        else:
            try:
                X = array(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        dataMatrix = mat(X)
        m  = dataMatrix.shape[0]
        aggClassEst = mat(zeros((m,1)))
        for i in range(len(self._adaboost)):
            classEst = self._stumpClassify(dataMatrix,self._adaboost[i]['dim'], \
                                           self._adaboost[i]['thresh'],\
                                           self._adaboost[i]['ineq'])
            aggClassEst += self._adaboost[i]['alpha']*classEst
            print(aggClassEst)
        return sign(aggClassEst)

class NotFittedError(Exception):
    """
    Exception class to raise if estimator is used before fitting

    """
    pass

if __name__ =="__main__":
    datMat=matrix([[1.,2.1],
                   [2.,1.1],
                   [1.3,1.],
                   [1.,1.],
                   [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    adaboost = Adaboost()
    adaboost.fit(datMat,classLabels)
    print(adaboost.predict([0,0]))
