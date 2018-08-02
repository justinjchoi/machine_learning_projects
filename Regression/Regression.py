# Machine Learning HW1
# Python 2.7x 
# run python Regression.py on the command line console 

import matplotlib.pyplot as plt
import numpy as np

# 1) open dataset 
def loadDataSet(filename):
    x = []
    y = []
    file = open(filename, 'r')
    for line in file: 
        line = line.rstrip()
        splitted = line.split('\t')
        x.append((float(splitted[0]), float(splitted[1])))
        y.append(float(splitted[2]))
    x_matrix = np.asmatrix(x)
    #plt.plot(x_matrix[:,1],y, 's', ms = 1)
    #plt.savefig("initial_loaded_data.png")
    return x, y

# 2) best line fit for linear regression 

# a) normal equation 
def standRegres(xVal, yVal):
    x_matrix = np.asmatrix(xVal)
    y_matrix = np.transpose(np.asmatrix(yVal))
    first_part = np.linalg.inv(np.transpose(x_matrix) * x_matrix)
    second_part = np.transpose(x_matrix) * y_matrix 
    theta = first_part * second_part
    l = np.arange(0.,1.1,.1)
    #plt.plot(np.asarray(x_matrix[:,1]), np.asarray(y_matrix), 's', l, l*theta[1,0] + theta[0,0], 'r--', ms = 1)
    #plt.title('y = ' + str(np.asscalar(theta[0])) + ' + ' + str(np.asscalar(theta[1])) + 'x')
    #plt.savefig("standRegres_Normal.png")
    return theta

# b) gradient descent 
    
def standRegres2(xVal,yVal):
    x_matrix = np.asmatrix(xVal)
    y_matrix = np.transpose(np.asmatrix(yVal))
    alpha = 0.05
    iterations = 10000
    theta=np.zeros((2,1))
    N=xVal.__len__()
    for i in range(iterations):
        prediction=x_matrix.dot(theta).flatten()
        temp=(yVal-prediction)
        theta[0][0]=theta[0][0]+alpha*(temp*x_matrix[:,0]).sum()/N
        theta[1][0]=theta[1][0]+alpha*(temp*x_matrix[:,1]).sum()/N
    y_matrix=theta[0][0]+theta[1][0]*x_matrix[:,1]
    #plt.plot(x_matrix[:,1],yVal, 's', ms = 1)
    #plt.plot(x_matrix,y_matrix,'r--', ms = 1)
    #plt.savefig("standRegres_GD.png")
    return theta

# 3) Polynomial Regression 

def polyRegres(xVal, yVal):
    x_matrix = np.asmatrix(xVal)
    y_matrix = np.transpose(np.asmatrix(yVal))
    x2 = np.empty((len(xVal),3))
    for i in range(0, len(xVal)):
        x2[i,0] = x_matrix[i,0]
        x2[i,1] = x_matrix[i,1]
        x2[i,2] = x_matrix[i,1] ** 2
    x_matrix = np.asmatrix(x2)
    first_part = np.linalg.inv(np.transpose(x_matrix) * x_matrix)
    second_part = np.transpose(x_matrix) * y_matrix 
    theta = first_part * second_part
    l = np.arange(0.,1.1,.1)
    #plt.plot(np.asarray(x_matrix[:,1]), np.asarray(y_matrix), 's', l, (l**2)*theta[2,0] + l*theta[1,0] + theta[0,0], 'r--', ms = 1)
    #plt.title('y = ' + str(np.asscalar(theta[0])) + ' + ' + str(np.asscalar(theta[1])) + 'x + ' + str(np.asscalar(theta[2])) + 'x^2')
    #plt.savefig("polyRegres.png")
    return theta


xVal, yVal = loadDataSet('/Users/choi/ML/HW1/Q2data.txt')
theta = standRegres(xVal, yVal)
ptheta = polyRegres(xVal, yVal)


## If you implement one more optimizatoin strategy  
theta2 = standRegres2(xVal, yVal)

