import numpy as np
import random
import math
from sklearn.cluster import KMeans

np.random.seed(1)
x = 2
data1 = np.random.normal(size=(100, 2)) + [ x, x]
data2 = np.random.normal(size=(100, 2)) + [ x,-x]
data3 = np.random.normal(size=(100, 2)) + [-x,-x]
data4 = np.random.normal(size=(100, 2)) + [-x, x]
data  = np.concatenate((data1, data2, data3, data4))
np.random.shuffle(data)
dataL = [0] * len(data)

center1 = random.choice(data)
center2 = random.choice(data)
center3 = random.choice(data)
center4 = random.choice(data)
centers = [center1, center2, center3, center4]

def euclideanD(x,y):
    return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5

changed = True
while changed:
    changed = False
    for i in range(len(data)):
        minD = math.inf
        tmp = dataL[i]
        for j in range(4):
            d = euclideanD(centers[j], data[i])
            if d < minD:
                dataL[i] = j + 1
                minD = d
        if dataL[i] != tmp:
            changed = True
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    for i,L in zip(range(len(data)),dataL):
        if L == 1:
            cluster1.append(data[i])
        if L == 2:
            cluster2.append(data[i])
        if L == 3:
            cluster3.append(data[i])
        if L == 4:
            cluster4.append(data[i])
    center1 = [sum(x[0] for x in cluster1)/len(cluster1),sum(x[1] for x in cluster1)/len(cluster1)]
    center2 = [sum(x[0] for x in cluster2)/len(cluster2),sum(x[1] for x in cluster2)/len(cluster2)]
    center3 = [sum(x[0] for x in cluster3)/len(cluster3),sum(x[1] for x in cluster3)/len(cluster3)]
    center4 = [sum(x[0] for x in cluster4)/len(cluster4),sum(x[1] for x in cluster4)/len(cluster4)]
    centers = [center1, center2, center3, center4]

X = data
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
print(dataL)
print(kmeans.labels_)
