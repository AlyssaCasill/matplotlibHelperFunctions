"""
Date: July 22, 2020
Written for the Gamble Lab @ Albert Einstein College of Medicine

Purpose: Functions to graph data using matplotlib
"""
__author__ =  'Alyssa D. Casill'
__version__ = '0.0.1'
__email__ = 'alyssa.casill@phd.einstein.yu.edu'

import matplotlib
from matplotlib import pyplot as plt
import palettable   
from palettable.cmocean.sequential import Thermal_20_r, Ice_20_r
from palettable.matplotlib import Viridis_20
import scipy.stats as ss
from scipy.stats import gaussian_kde
import numpy as np

def histogram(data,plotPath,log="Y",frac="N"):
    """
    Creates a histogram of input data.

    Input: 
    Lists of floats
    Full path to output location (including png/jpg/pdf/etc file extension)
    
    Options: 
    log (if Y, data will be transformed using np.log2)
    frac (if Y, the result is the value of the probability density function at the bin, normalized such that the integral over the range is 1.) ***THIS DOESN'T WORK***
    
    TO UPDATE:
    Fix density function
    More options for labels/titles
    """
    if log == "Y":
        data = [np.log2(i) for i in data] 
    if frac == "N":
        datahist,databins = np.histogram(data,bins="auto")
    else:
        datahist,databins = np.histogram(data,bins="auto",density=True)
    width = 0.7 * (databins[1]-databins[0])
    center = (databins[:-1] + databins[1:]) / 2
    plt.bar(center, datahist, align='center', width=width, color="black", edgecolor="black")
    plt.savefig(plotPath)
    plt.close()

def boxPlot(dataDict,plotPath,xname,yname,log):
    """
    Creates a box plot of input data.

    Input: 
    Data dictionary: dataDict[group]=[list of values in group]
    Full path to output location (including png/jpg/pdf/etc file extension)
    
    Options: 
    xname, yname (strings to be printed as axis labels)
    log (if Y, data will be transformed using np.log2)
    
    TO UPDATE:
    Option for graph title
    """
    if log == "Y":
        for k,v in dataDict.items():
            dataDict[k] = [np.log2(i) for i in v]
    sortedKeys = sorted(dataDict.keys())
    labels = []
    data = []
    for i in sortedKeys:
        labels.append(i)
        data.append(dataDict[i])
    labels = tuple(labels)
    data = tuple(data)
    flyers = dict(markerfacecolor='black', marker='.', markersize="1")
    median = dict(linestyle='-', linewidth=1, color='black')
    plt.boxplot(data,flierprops=flyers,medianprops=median)
    plt.xticks(range(1,len(labels)+1), labels, rotation=45)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.tight_layout()
    plt.savefig(plotPath)
    plt.close()

def CDF(dataDict,plotPath,xname,title="",log="N",c=Viridis_20.mpl_colormap):
    """
    Creates a CDF of input data.

    Input: 
    Data dictionary: dataDict[group]=[list of values in group]
    Full path to output location (including png/jpg/pdf/etc file extension)
    
    Options: 
    xname (string to be printed as axis labels - should be the variable that data is grouped by)
    title (string for graph title)
    log (if Y, data will be transformed using np.log2)
    c (colormap)

    TO UPDATE:
    """
    if log == "Y":
        for k,v in dataDict.items():
            dataDict[k] = [np.log2(i) for i in v]
    labels = sorted(dataDict.keys())
    cmap = plt.get_cmap(c)
    N = len(labels)
    for k in labels:
        x = sorted(dataDict[k])
        y = [(x.index(i)+1)/len(x) for i in x]
        color = cmap(float(labels.index(k))/N)
        plt.plot(x,y,c=color,label=k+" n="+str(len(dataDict[k])))
    plt.legend()
    plt.title(title)
    plt.xlabel(xname)
    plt.ylabel("CDF")
    plt.savefig(plotPath)
    plt.close() 

def kdeScatterPlot(dataA,dataB,plotPath,xname="",yname="",log="Y",logaxis="both",identity="Y",c=Thermal_20_r.mpl_colormap):
    """
    Creates a scatterplot colored based on the density of points in an area, calcluated by the Kernel Density Estimation.

    Input: 
    Two lists of floats (x, y matched values, must be equal length)
    Full path to output location (including png/jpg/pdf/etc file extension)

    Options: 
    xname, yname (strings to be printed as axis labels)
    log (if Y, data will be transformed using np.log2 based on the logaxis argument)
    logAxis (if both, x and y data will undergo log transformation, if X, only X values will be logged, if Y, only Y values will be logged)
    Identity (if Y, a line of identity will be drawn based on the minimum and maximum values in the dataset)
    c (colormap)
    
    TO UPDATE:
    Log and identity options should be True or False instead of Y/N
    Easier to manipulate graph title?
    """
    
    if log == "Y":
        if logaxis == "both":
            dataA = [np.log2(i) for i in dataA]
            dataB = [np.log2(i) for i in dataB]
        elif logaxis == "X":
            dataA = [np.log2(i) for i in dataB]
        elif logaxis == "Y":
            dataB = [np.log2(i) for i in dataB]
    top = max(max(dataA),max(dataB))
    bottom = min(min(dataA),min(dataB))
    x = [bottom,top]
    y = [bottom,top]
    corr,pval = ss.spearmanr(dataA,dataB)
    print(len(dataA))
    print(corr)
    print(pval)
    xy = np.vstack([dataA,dataB])
    z = gaussian_kde(xy)(xy)
    fig, ax = plt.subplots(1,1,figsize=(7.5,6))
    cax = ax.scatter(dataA,dataB,c=z,s=10,edgecolor='',marker="o",zorder=2,cmap=c)
    fig.colorbar(cax)
    if identity == "Y":
        plt.plot(x,y,linestyle="-",linewidth=1,color="black",zorder=1)
    ax.margins(x=0)
    ax.margins(y=0)
    # ax.grid()
    if log == "Y":
        if logaxis == "both":
            x = []
            y = []
            for i in ax.get_xticks():
                l = 2**i
                if l < 0.25:
                    x.append(format(l,'.4f'))
                else:
                    x.append(l)
            for i in ax.get_yticks():
                l = 2**i
                if l < 0.25:
                    y.append(format(l,'.4f'))
                else:
                    y.append(l)
            ax.set_xticklabels(x)
            ax.set_yticklabels(y)
        elif logaxis == "X":
            x = []
            for i in ax.get_xticks():
                l = 2**i
                if l < 0.25:
                    x.append(format(l,'.4f'))
                else:
                    x.append(l)
            ax.set_xticklabels(x)
        elif logaxis == "Y":
            y = []
            for i in ax.get_yticks():
                l = 2**i
                if l < 0.25:
                    y.append(format(l,'.4f'))
                else:
                    y.append(l)
            ax.set_yticklabels(y)
    plt.title("r="+str(corr)+",p="+str(pval)+",n="+str(len(dataA))+"\n"+plotPath.split("\\")[-1])
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(plotPath)
    plt.close()   

def ViolinPlot(dataDict,plotPath,xname,yname,log):
    """
    Creates a violin plot of input data.

    Input: 
    Data dictionary: dataDict[group]=[list of values in group]
    Full path to output location (including png/jpg/pdf/etc file extension)
    
    Options: 
    xname, yname (string to be printed as axis labels)
    title (string for graph title)
    log (if Y, data will be transformed using np.log2)

    TO UPDATE:
    Improve graph appearance
    """
    if log == "Y":
        for k,v in dataDict.items():
            dataDict[k] = [np.log2(i) for i in v]
    sortedKeys = sorted(dataDict.keys())
    labels = []
    data = []
    for i in sortedKeys:
        labels.append(i)
        data.append(dataDict[i])
    labels = tuple(labels)
    data = tuple(data)
    plt.violinplot(data,showmeans=True,showextrema=False)
    plt.xticks(range(1,len(labels)+1), labels, rotation=45)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.tight_layout()
    plt.savefig(plotPath)
    plt.close()


if __name__ == "__main__":
    from numpy import random

    testDataX = random.rand(50)
    testDataY = random.rand(50)


