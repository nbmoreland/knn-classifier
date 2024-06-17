# Nicholas Moreland
# 1001886051

import numpy as np
import math

# Normalize the training and test data
def normalize(training_data, test_data):
    means = np.mean(training_data, axis=0)
    stds = np.std(training_data, axis=0, ddof=1)

    for i in range(len(stds)-1):
        if stds[i] == 0:
            stds[i] = 1

    for i in range(len(training_data)):
        for j in range(len(means) - 1):
            training_data[i][j] = (training_data[i][j] - means[j]) / stds[j]

    for i in range(len(test_data)):
        for j in range(len(means) - 1):
            test_data[i][j] = (test_data[i][j] - means[j]) / stds[j]

# Calculates the distance between two vectors
def summation(v1, v2):
    summation = 0
    for i in range(len(v1) - 1):
        summation += (v1[i] - v2[i]) ** 2

    return math.sqrt(summation)

# Sorts the list of distances
def sort(sub_li): 
    sub_li.sort(key = lambda x: x[0]) 
    return sub_li

# KNN algorithm
def knn_classify(training_file, test_file, k):
    # Read data from file
    file = open(training_file, 'r')
    lines = file.readlines()

    # Load training data and classes
    classes = []
    training_data = []
    for row in lines:
        temp = row.split()
        training_data.append(np.array(temp).astype(np.float64))
        if int(temp[-1]) not in classes:
            classes.append(int(temp[-1]))

    # Read data from file
    file = open(test_file, 'r')
    lines = file.readlines()

    # Load test data
    test_data = []
    for row in lines:
        temp = row.split()
        test_data.append(np.array(temp).astype(np.float64))

    # Normalize the data
    normalize(training_data, test_data)

    # Classify the test data
    total_accur = 0
    for n in range(len(test_data)):
        # Calculate the distance between the test data and training data
        distances = []
        for i in range(len(training_data)):
            distances.append([summation(training_data[i], test_data[n]), training_data[i][-1]])

        # Sort the distances
        sort(distances)

        # Find the k nearest neighbors
        k_nearest_class = []
        for i in range(k):
            k_nearest_class.append(int(distances[i][1]))

        # Find the most common class
        max_count = 0
        predicted_class = [0]
        for i in classes:
            counts = k_nearest_class.count(i)
            if counts > max_count:
                max_count = counts
                predicted_class.clear()
                predicted_class.append(i)
            elif counts == max_count:
                predicted_class.append(i)

        # Calculate the accuracy
        accuracy = 1 / len(predicted_class) if int(test_data[n][-1] in predicted_class) else 0
        total_accur += accuracy
        print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(n+1, int(predicted_class[0]), int(test_data[n][-1]), accuracy))
    
    # Print the classification accuracy
    print("classification accuracy={:6.4f}".format(total_accur/len(test_data)))