import numpy as np
import random
import operator
import math
import time

df_full = []
columns = []
features = columns[:len(columns) - 1]
class_labels = []
df = []

# Number of Attributes
num_attr = 0

# Number of Clusters
k = 2

# Maximum number of iterations
MAX_ITER = 100

# Number of data points
n = len(df)

# Fuzzy parameter
m = 2.00

class ExistingFCM:

    def initializeMembershipMatrix(self):
        membership_mat = list()
        for i in range(n):
            random_num_list = [random.random() for i in range(k)]
            summation = sum(random_num_list)
            temp_list = [x / summation for x in random_num_list]
            membership_mat.append(temp_list)
        return membership_mat

    def calculateClusterCenter(self, membership_mat):
        cluster_mem_val = zip(*membership_mat)
        cluster_centers = list()
        for j in range(k):
            x = list(cluster_mem_val[j])
            xraised = [e ** m for e in x]
            denominator = sum(xraised)
            temp_num = list()
            for i in range(n):
                data_point = list(df.iloc[i])
                prod = [xraised[i] * val for val in data_point]
                temp_num.append(prod)
            numerator = map(sum, zip(*temp_num))
            center = [z / denominator for z in numerator]
            cluster_centers.append(center)
        return cluster_centers

    def updateMembershipValue(self, membership_mat, cluster_centers):
        p = float(2 / (m - 1))
        for i in range(n):
            x = list(df.iloc[i])
            distances = [np.linalg.norm(map(operator.sub, x, cluster_centers[j])) for j in range(k)]
            for j in range(k):
                den = sum([math.pow(float(distances[j] / distances[c]), p) for c in range(k)])
                membership_mat[i][j] = float(1 / den)
        return membership_mat

    def getClusters(self, membership_mat):
        cluster_labels = list()
        for i in range(n):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
            cluster_labels.append(idx)
        return cluster_labels

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
        time.sleep(9)
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