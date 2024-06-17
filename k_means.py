# Nicholas Moreland
# 1001886051

import numpy as np
import math
import random

# Calculates the distance between two vectors
def summation(v1, v2):
    summation = 0
    for i in range(len(v1)):
        summation += (v1[i] - v2[i])**2
    return math.sqrt(summation)

# Groups the data by cluster
def group_data(vals):
    values = set(map(lambda x: x[1], vals))
    return [[y[0] for y in vals if y[1] == x] for x in values]

# K-means algorithm
def k_means(data_file, K, initialization):
    file = open(data_file, 'r')
    lines = file.readlines()

    # Read data from file
    data = []
    for line in lines:
        temp = line.split()
        temp = [float(x) for x in temp]
        data.append([temp, -1])

    # Initialize clusters depending on initialization type
    if initialization == 'random':
        for i in range(len(data)):
            data[i][1] = random.randint(0, K-1)
    elif initialization == 'round_robin':
        cluster = 0
        for i in range(len(data)):
            data[i][1] = cluster
            cluster = (cluster + 1) % K
    grouped_data = group_data(data)

    # Calculate cluster means
    cluster_means = []
    for i in range(K):
        cluster_means.append(np.mean(grouped_data[i], axis=0))

    # Iterate until clusters do not change
    while True:
        cluster_new = []
        for i in range(len(data)):
            cluster = 0
            smallest_dist = 99999

            for x in range(len(cluster_means)):
                if summation(data[i][0], cluster_means[x]) < smallest_dist:
                    cluster = x
                    smallest_dist = summation(data[i][0], cluster_means[x])
            cluster_new.append([data[i][0], cluster])
        
        clusters_same = True
        for i in range(len(data)):
            if data[i][1] != cluster_new[i][1]:
                clusters_same = False
                break
        if clusters_same == True:
            break

        data = cluster_new.copy()
        grouped_data = group_data(data)

        for i in range(K):
            cluster_means[i] = np.mean(grouped_data[i], axis=0)

    # Print results
    for i in range(len(data)):
        if len(data[i][0]) == 1:
            print("{:10.4f} --> cluster {:d}".format(data[i][0][0], data[i][1]+1))
        else:
            print("({:10.4f}, {:10.4f}) --> cluster {:d}".format(data[i][0][0], data[i][0][1], data[i][1]+1))