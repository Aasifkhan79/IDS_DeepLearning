import random
import time
from sklearn.datasets import make_blobs
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import Birch
from typing import Tuple, Dict, List

class ExistingBirch:
    def clusterbirch(self):
        # from sklearn.datasets.samples_generator import make_blobs
        from sklearn.cluster import Birch

        X, clusters = make_blobs(n_samples=450, centers=6, cluster_std=0.70, random_state=0)

        brc = Birch(branching_factor=50, n_clusters=None, threshold=1.5)
        brc.fit(X)

        labels = brc.predict(X)


        # initialize the data set we'll work with
        training_data, _ = make_classification(
            n_samples=1000,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=4
        )

        # define the model
        birch_model = Birch(threshold=0.03, n_clusters=2)

        # train the model
        birch_model.fit(training_data)

        # assign each data point to a cluster
        birch_result = birch_model.predict(training_data)

        # get all of the unique clusters
        birch_clusters = unique(birch_result)

        # plot the BIRCH clusters
        for birch_cluster in birch_clusters:
            # get data points that fall in this cluster
            index = where(birch_result == birch_clusters)
            # make the plot
            pyplot.scatter(training_data[index, 0], training_data[index, 1])

        # show the BIRCH plot
        pyplot.show()

    def load_data(self, file_name) -> List[List]:
        print("--->Loading csv file")

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            data = []

            for line in csv_reader:
                if line_count == 0:
                    print(f'Column names: [{", ".join(line)}]')
                else:
                    data.append(line)
                line_count += 1

        print(f'Loaded {line_count} records')
        return data

    def compute_clusters(self, data: List) -> np.ndarray:
        print("--->Computing clusters")
        birch = Birch(
            branching_factor=50,
            n_clusters=5,
            threshold=0.3,
            copy=True,
            compute_labels=True
        )

        birch.fit(data)
        predictions = np.array(birch.predict(data))
        return predictions

    def show_results(self, data: np.ndarray, labels: np.ndarray, plot_handler="seaborn") -> None:
        labels = np.reshape(labels, (1, labels.size))
        data = np.concatenate((data, labels.T), axis=1)

        # Seaborn plot
        if plot_handler == "seaborn":
            facet = sns.lmplot(
                data=pd.DataFrame(data, columns=["Income", "Spending", "Label"]),
                x="Income",
                y="Spending",
                hue='Label',
                fit_reg=False,
                legend=True,
                legend_out=True
            )

        # Pure matplotlib plot
        if plot_handler == "matplotlib":
            fig = plt.figure()
            ax = fig.add_subplot(111)
            scatter = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=50)
            ax.set_title("Clusters")
            ax.set_xlabel("Income")
            ax.set_ylabel("Spending")
            plt.colorbar(scatter)
        plt.show()

    def show_data_corelation(self, data=None, csv_file_name=None):
        data_set = None
        if csv_file_name is None:
            cor = np.corrcoef(data)
            print("Corelation matrix:")
            print(cor)
        else:
            data_set = pd.read_csv(csv_file_name)
            print(data_set.describe())
            data_set = data_set[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
            cor = data_set.corr()
        sns.heatmap(cor, square=True)
        plt.show()
        return data_set

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
        time.sleep(18)
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

