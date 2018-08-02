# Machine Learning HW2-Ridge

__author__ = 'Justin Choi (jc8mc)'

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import axes3d

def loadDataSet(filename):
    data=np.loadtxt(filename)
    xVal=data[:,0:3]
    yVal=data[:,3]
    return xVal,yVal

def ridgeRegress(xVal,yVal,lbd):
    x_mtx=np.matrix(xVal)
    y_mtx=np.matrix(yVal)
    betaLR=(x_mtx.getT()*x_mtx+np.identity(3)*lbd).getI()*x_mtx.getT()*y_mtx.getT()
    betaLR[0]=sum(yVal)/len(xVal)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(xVal[:,1],xVal[:,2],yVal,c='r',marker='o')
    x1=np.arange(min(xVal[:,1]),max(xVal[:,1]),(max(xVal[:,1])-min(xVal[:,1]))/20)
    x2=np.arange(min(xVal[:,2]),max(xVal[:,2]),(max(xVal[:,2])-min(xVal[:,2]))/20)
    y=betaLR[0]+np.multiply(betaLR[1],x1)+np.multiply(betaLR[2],x2)
    X1,X2=np.meshgrid(x1,x2)
    ax.plot_wireframe(X1,X2,y,rstride=1,cstride=1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.savefig('ridge_lambda0.png')
    return betaLR

def genTrainTestData(dict,xVal,yVal,idx):
    foldVol=int(len(xVal)/10)
    localTestIdx=range(foldVol*idx,foldVol*(idx+1))
    localAllIdx=range(len(xVal))
    localTrainIdx=np.delete(localAllIdx,localTestIdx)
    testIdx=dict[localTestIdx]
    trainIdx=dict[localTrainIdx]
    return xVal[trainIdx],yVal[trainIdx],xVal[testIdx],yVal[testIdx]

def test(xVal,yVal,betaLR):
    y=betaLR[0]+np.multiply(betaLR[1],xVal[:,1])+np.multiply(betaLR[2],xVal[:,2])
    error=(yVal-y)*(yVal-y).getT()
    return error

def cv(xVal,yVal):
    MSE=[]
    numLbd=50
    numFold=10
    J = 0.0
    random.seed(37)
    dict=list(range(len(xVal)))
    random.shuffle(dict)
    dict=np.asarray(dict)
    for i in range(1,numLbd+1):
        J = 0.0
        for j in range(numFold):
            xTr,yTr,xTe,yTe=genTrainTestData(dict,xVal,yVal,j)
            betaLR=ridgeRegress(xTr,yTr,i*0.02)
            error=test(xTe,yTe,betaLR)
            c=np.asarray(error)
            J+=c[0]
        J=J/10.0
        MSE.append(J)
    MSE=np.asarray(MSE)
    lbd=np.linspace(0.02,1,num=50)
    fig=plt.plot(lbd,MSE,'r-')
    plt.xlabel('lambda')
    plt.ylabel('10-fold cv J')
    plt.show()
    index=np.where(MSE==MSE.min())
    lambdaBest=float(index[0][0]+1)*0.02
    return lambdaBest


def standRegres(xVal):
    x_mtx=np.matrix(xVal[:,0:2])
    y_mtx=np.matrix(xVal[:,2])
    theta=(x_mtx.getT()*x_mtx).getI()*x_mtx.getT()*y_mtx.getT()
    plt.plot(xVal[:,1],xVal[:,2],'ro',x_mtx[:,1],theta[0]+np.multiply(theta[1],x_mtx[:,1]),'b-',linewidth=2.0)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig('stdregres_noridge.png')
    return theta


if __name__ == "__main__":
    xVal,yVal=loadDataSet("RRdata.txt")
    betaLR=ridgeRegress(xVal,yVal,0)
    print(betaLR)
    lambdaBest=cv(xVal,yVal)
    print(lambdaBest)
    theta=standRegres(xVal)
    print(theta)
    betaRR=ridgeRegress(xVal,yVal, lambdaBest)
    print(betaRR)
