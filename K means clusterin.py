import numpy as np
from numpy import genfromtxt
import random
import sys


def findloss(cluster,datapoints,assignedPoints,cent):
    """This function calculates the  loss for particular cluster
            Args:
                Clusters centers, data  points, cluster number and cluster-data membership.
            Returns:  Loss
    """
    loss=0
    for i in range(datapoints.shape[0]):
        if assignedPoints[i]==cent:
            loss+=np.linalg.norm(datapoints[i] - cluster)**2
    return loss

def findcluster(number,datapoints,assignedPoints,cent):
    """This function calculates the cluster centers
               Args:
                   datapoints, cluster-data membership,cluster number and number of datapoints cluster has
               Returns:  cluster
    """
    temp=np.zeros((1,7))
    for i in range(datapoints.shape[0]):
        if assignedPoints[i]==cent:
            temp=np.add(temp,datapoints[i])
    temp=temp/number
    return temp

def assigncentroid(center,datapoint):
    """This function finds center for passed data points
                  Args:
                      datapoints and cluster centers
                  Returns:  cluster center
       """
    c=0
    min=sys.maxint
    for cent in range(center.shape[0]):
        if(np.linalg.norm(center[cent] - datapoint)<min):
            min=np.linalg.norm(center[cent] - datapoint)
            c=cent
    return c


if __name__=="__main__":
    data = genfromtxt('seed.txt', delimiter=',')
    data=data[:,0:data.shape[1]-1]
    dim=7
    karray=[2,3,4,5,6,7,8,9,10]
    objective_loss=np.zeros(9)

    #loop to run the K-means clustering five times
    for iteration in range(5):
        for k in karray:
            cluster=np.zeros((k, dim))
            for i in range(cluster.shape[0]):
                cluster[i]=data[random.randint(0,data.shape[0]-1)]
            prevloss=sys.maxint
            clusterNumber=np.zeros(k)
            clusterAssign={}
            while True:
                clusterNumber = np.zeros(k)
                #loop to assign cluste center for each data point
                for i in range(data.shape[0]):
                    centroid=assigncentroid(cluster,data[i])
                    clusterAssign[i]=centroid
                    clusterNumber[centroid]+=1

                #loop to split the maximum cluster if any of the cluster has zero datapoints in it
                for indx,num in enumerate(clusterNumber):
                    if num==0:
                        max=np.argmax(clusterNumber)
                        clusterNumber[indx]=clusterNumber[max]/2
                        clusterNumber[max]=clusterNumber[max]/2
                        count=0
                        while count<clusterNumber[indx]:
                            rand=random.randint(0, data.shape[0] - 1)
                            if clusterAssign[rand]==max:
                                clusterAssign[rand]=indx
                                count+=1

                #loop to calculate the cluster center
                for i in range(cluster.shape[0]):
                    cluster[i]=findcluster(clusterNumber[i],data,clusterAssign,i)
                currentloss=0

                #loop to calculate the objective function
                for i in range(cluster.shape[0]):
                    currentloss+=findloss(cluster[i],data, clusterAssign,i)

                if currentloss==prevloss:
                    break
                if currentloss<1300:
                    if currentloss>prevloss:
                        break
                prevloss=currentloss
            objective_loss[k-2]+=prevloss
    objective_loss=np.divide(objective_loss,5.0)
    for indx, loss in enumerate(objective_loss):
        print "Objective Function for "+repr(indx+2)+" Cluster is "+ repr(loss)







