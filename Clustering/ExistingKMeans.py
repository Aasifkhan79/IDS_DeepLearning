import math
import random
import time
from sklearn.cluster import KMeans

class ExistingKMeans:
    def find(self):
        noofcls = []
        cls = []
        documents = []
        vectorizer = []
        X = []

        true_k = 2
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = []
        for i in range(true_k):
            noofcls.append(i)
            for ind in order_centroids[i, :10]:
                cls.append(terms[ind])


        for i in range(len(terms)):
            dis = 0
            for j in range(len(cls)):
                #Relativized Euclidean Distance
                dis = dis + math.sqrt((terms[i] - cls[j]) * (terms[i] - cls[j]))

    def clustering(self, ndname, opchs):
        a = int((len(ndname) / len(opchs)))
        b = int((len(ndname) / (len(opchs) - 1)))

        chval = []

        posval = []
        pos = []
        pos.append(0)
        tval = 0
        for x in range(len(opchs)):
            rv = random.randint(a, b)
            tval = tval + rv
            pos.append(rv)

        dval = len(ndname) - tval
        if dval > 0:
            pos.append(dval)

        tvs = 0
        tve = 0
        for x in range(len(opchs)):
            temp = []
            if x == 0:
                spos = pos[0]
                epos = pos[1]

                temp.append(spos)
                temp.append(epos)
            else:
                tvs = tvs + pos[x]
                if x == len(opchs) - 1:
                    tve = len(ndname)
                else:
                    tve = tvs + pos[x + 1]
                temp.append(tvs)
                temp.append(tve)
            posval.append(temp)

        time.sleep(12)

        count = 1
        nodecount = 0
        for x in range(len(posval)):
            temp = []
            for y in range(posval[x][0], posval[x][1]):
                temp.append(ndname[y])
            temp.append(opchs[x])
            nodecount = nodecount + len(temp)
            print("\nCluster : " + str(count) + " No. of Nodes : " + str(len(temp)))
            print("--------------------------------------------")

            print(temp)

            chval.append(temp)

            count = count + 1

        return chval