import pandas as pd
import matplotlib.pyplot as plt
from decimal import *
import numpy
from numpy import *
from web import iters

def normalEquation(df):
    x = df.iloc[:, :-1]
    x.insert(0, 'feature 0', 1)

    y = df.iloc[:, -1:]

    x=numpy.array(x)
    y=numpy.array(y)

    theetas=numpy.dot( numpy.dot( numpy.linalg.inv( numpy.dot(numpy.transpose(x), x)) ,numpy.transpose(x)), y)

    t=[]
    for a in theetas:
        for b in a:
            t.append(b)
    return t


def plotItervsCost(iter,costs):

    plt.plot(iter,costs)
    plt.xlabel('No of iterations')
    plt.ylabel('cost J')
    plt.title('Figure: Convergence of gradient descent with an appropriate learning rate')
    plt.show()

def getMean(df):
    means=[1.0001]
    df = df.iloc[:, :-1]

    for column in df:
        means.append(df[column].mean())

    return means
#############################
def getStd(df):

    stds = [0.0001]
    df = df.iloc[:, :-1]

    for column in df:
        stds.append(df[column].std())

    return stds
##############################
def get_theetas_list(d):
    no_of_theetas = max(len(d[x]) for x in range(len(d)))
    theetas = []
    for i in range(no_of_theetas):
        theetas.append(0.0)
    return theetas
##########################################
def cost( dataList, theetas,means,stds):
    c=0.0

    features = [1.0]

    for row in dataList:
        for i in range(len(row)-1):
            features.append(row[i])

        c+= float( format((linearHypothesis(features,theetas,means,stds)-row[len(row)-1])** 2,'.10f'))
        features = [1.0]

    c=c/(2*len(dataList))
    return c
##########################################

def derivative_of_cost( dataList, theetas, theeta_index,means,stds):
    c=0.0
    features = [1.0]

    for row in dataList:
        for i in range(len(row)-1):
            features.append(row[i])

        c += (linearHypothesis(features, theetas,means,stds) - row[len(row) - 1]) * ((features[theeta_index]-means[theeta_index])/stds[theeta_index])
        features = [1.0]

    c=c/(len(dataList))
    return c
##########################################


def linearHypothesis (features,theetas,means,stds):
    returnValue=0.0
    for f,t,m,s in zip(features,theetas,means,stds):
        returnValue+=((f-m)/s)*t
    return returnValue

def linearHypothesisWithoutNormalization (features,theetas):
    returnValue=0.0
    for f,t in zip(features,theetas):
        returnValue+=f*t
    return returnValue


###########################################

def linearRegression(d,means,stds,iterations,alpha):

    theetas = get_theetas_list(d)
    tempTheetas=[ 0.0 for x in theetas]
    costs = []
    iter = []

    for i in range(iterations):
        for j in range(len(theetas)):
            tempTheetas[j]=theetas[j]-(alpha*derivative_of_cost(d,theetas,j,means,stds))
        theetas=tempTheetas[:]
        #print(theetas)
        #print(cost(d,theetas,means,stds))
        costs.append(cost(d, theetas, means, stds))
        iter.append(i+1)

    plotItervsCost(iter, costs)

    return theetas

def dataPlot():
    d = pd.read_csv('ex1data1.txt', names=['population','profit'])
    d.plot(kind='scatter',x='population', y='profit', color='red', marker='x', label='Training data')
    plt.title('Figure 1: Scatter plot of training data ')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

def dataPlotAfterLearning(theetas,means,stds):

    d = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
    d.plot(kind='scatter', x='population', y='profit', color='red', marker='x',label='Training data')
    plt.title('Figure 1: Scatter plot of training data ')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')

    x= d.population.tolist();
    x = numpy.array(x)
    plt.plot(x,linearHypothesis([1.0,x],theetas,means,stds), label='Linear regression')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    alpha=0.03

    #for file extdata1.txt
    dataPlot()
    df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
    means = getMean(df)
    stds = getStd(df)
    d = df.values.tolist()

    theetas=linearRegression(d, means,stds,1500,alpha)
    dataPlotAfterLearning(theetas,means,stds)

    print(theetas)
    print(cost(d, theetas,means,stds))


    #################################################
    #for file ex1data2.txt

    df = pd.read_csv('ex1data2.txt', names=['size', 'no_of_bedrooms','price'])
    means = getMean(df)
    stds = getStd(df)
    d = df.values.tolist()

    theetas = linearRegression(d,means,stds,1500,alpha)
    #print(theetas)
    #print(cost(d, theetas, means, stds))
    print(linearHypothesis([1, 1650, 3], theetas, means, stds))

    ###################################################
    df = pd.read_csv('ex1data2.txt', names=['size', 'no_of_bedrooms', 'price'])
    theetas=normalEquation(df)
    print(linearHypothesisWithoutNormalization([1,1650,3], theetas))






