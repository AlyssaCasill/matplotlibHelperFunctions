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
import itertools
import math
import scipy.stats as ss
from scipy.stats import gaussian_kde
import numpy as np

"""
SECTION 1
These functions can be used to manipulate data into a format that works with matplotlibHelperFunctions

Potential updates:
Make dataDict a class? So that it would know how to make itself or read one in and perform calculations/make graphs?
"""
                
def createGroupedLists(dataA,dataB,num):
    """
    Groups data A based on data B.

    Input: 
    Two lists of floats: dataA will be broken up into groups based on the status of the matched value in dataB
    num = number of groups

    Return:
    List of tuples denoting the edges of each group
    List of lists of values in each group (from Data A)
    List of values in each group (from Data B)
    """
    num = int(num)
    zipped2 = sorted([(i,j) for i,j in zip(dataB,dataA)])
#    zipped2.sort()
    zipped = [i for i in zipped2 if not math.isnan(i[1])]
    div = int(len(zipped)/num)
    varSorted = [i[0] for i in zipped]
    zippedSorted = [i[1] for i in zipped]
    bounds1 = [i for i in range(0,len(zippedSorted)+1,div)][:-1]
    bounds2 = [i for i in range(0,len(zippedSorted)+1,div)][1:-1]
    bounds2.append(len(zippedSorted))
    edges = []
    for i,j in zip(bounds1,bounds2):
        edges.append((varSorted[i],varSorted[j-1]))
    lists = []
    groups = []
    for i,j in zip(bounds1,bounds2):
        lists.append(zippedSorted[i:j])
        groups.append(varSorted[i:j])
    return(edges,lists,groups)

def xyDataDict(dataA,dataB,n,output="groupedData"):
    """
    Creates dataDict based on two lists of data. Data A will be put into groups keyed by ranges of Data B.

    Input: 
    Two lists of floats: dataA will be broken up into groups based on the status of the matched value in dataB
    n = number of groups

    Options:
    output (if groupedData, will return dictionary keyed by group bounds where the values are lists of data in those groups. if not, will return dictionary keyed by group bound where the values are the lists of data making up the groups, originally used to split up the data)
    """
    edges,lists,groups = createGroupedLists(dataA,dataB,n)
    dataDict = dict()
    if output == "groupedData":
        for bounds,data in zip(edges,lists):
            dataDict[bounds] = data 
        return(dataDict)
    else:
        for bounds,data in zip(edges,groups):
            dataDict[bounds] = data 
        return(dataDict)

"""
SECTION 2
These functions work with the dataDict data structure to calculate statistics and output data
"""

def groupStats(dataDict,test="mwu"):
    """
    Calculates statistical significance of differences between all pairings within a dataDict. Currently uses either a Mann-Whitney U test or a KS test.

    Input: 
    Data dictionary: dataDict[group]=[list of values in group]
    test (either mwu or ks)

    Return:
    Dictionary [(groupA,groupB)] = [mean1,median1,n1,mean2,median2,n2,pval]
    """
    ttest = dict()
    for x in dataDict.keys():
        for y in dataDict.keys():
            if x!=y and not ((x,y) in ttest.keys()) and not ((y,x) in ttest.keys()):
                mean1 = np.mean(dataDict[x])
                mean2 = np.mean(dataDict[y])
                median1 = np.median(dataDict[x])
                median2 = np.median(dataDict[y])
                n1 = len(dataDict[x])
                n2 = len(dataDict[y])
                if test == "mwu":
                    pval = ss.mannwhitneyu(dataDict[x],dataDict[y])[1]
                elif test == "ks":
                    pval = ss.ks_2samp(dataDict[x],dataDict[y],mode='asymp')[1]
                ttest[(x,y)]=[mean1,median1,n1,mean2,median2,n2,pval]
    return(ttest)

def groupStatsDataOut(outfileName,statsDict):
    """
    Prints statistics dictionary to file.

    Input: 
    outfileName (path to output file)
    statsDict (statistics dictionary)
    """
    outfile = open(outfileName,"w")
    outfile.write("pair\tmean1\tmedian1\tn1\tmean2\tmedian2\tn2\tmwpval\n")
    for k,v in statsDict.items():
        outfile.write(str(k)+"\t"+"\t".join([str(i) for i in v])+"\n")
    outfile.close()

def dataOut(outfileName,dataDict):
    """
    Prints data dictionary to file (optimized for use in graphpad prism).

    Input: 
    outfileName (path to output file)
    dataDict (data dictionary)
    """
    outfile = open(outfileName,"w")
    sortedKeys = sorted(dataDict.keys())
    labels = []
    data = []
    for i in sortedKeys:
        labels.append(i)
        data.append(dataDict[i])
    outfile.write("\t".join([str(i) for i in labels]))
    outfile.write("\n")
    zipped = [i for i in itertools.zip_longest(*data,fillvalue="")]
    zipped2 = [list(i) for i in zipped]
    for x in zipped2:
        xx = []
        for i in x:
            try:
                xx.append(str(i))
            except ValueError:
                xx.append("")
        outfile.write("\t".join(xx)+"\n")

"""
SECTION 3
These functions make graphs of input data.
"""

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
        plt.plot(x,y,c=color,label=str(k)+" n="+str(len(dataDict[k])))
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

def violinPlot(dataDict,plotPath,xname,yname,log):
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

    testDataX = random.rand(2000)
    testDataY = random.rand(2000)

    histogram(testDataX,"C:\\Users\\acasill\\Desktop\\graphTests\\example_histogram.png",log="N")

    kdeScatterPlot(testDataX,testDataY,"C:\\Users\\acasill\\Desktop\\graphTests\\example_kdeScatter.png","x","y",log="N")

    dataDict = xyDataDict(testDataX,testDataY,5)

    CDF(dataDict,"C:\\Users\\acasill\\Desktop\\graphTests\\example_cdf.png","x","example")

    boxPlot(dataDict,"C:\\Users\\acasill\\Desktop\\graphTests\\example_boxPlot.png","X","Y",log="N")

    violinPlot(dataDict,"C:\\Users\\acasill\\Desktop\\graphTests\\example_violinplot.png","x","y",log="N")

    dataOut("C:\\Users\\acasill\\Desktop\\graphTests\\example_data.txt",dataDict)

    ks = groupStats(dataDict,test="ks")
    groupStatsDataOut("C:\\Users\\acasill\\Desktop\\graphTests\\example_ks_stats.txt",ks)





