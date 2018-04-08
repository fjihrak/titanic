import psycopg2
import numpy as np
import csv

def open_with_python_csv(filename):
    data = []
    with open(filename, 'r') as filename:
        reader = csv.reader(filename)
        header = next(reader)
        for row in reader:
            data.append(row)
    data = np.array(data)

    return header, data

def label_from_csv(filename):
    data = []
    with open(filename, 'r') as filename:
        reader = csv.reader(filename)
        header = next(reader)
        for row in reader:
            data.append(list(map(int,row)))

    data = np.array(data)
    # label = list(data)[1:,1:]
    label = data.T[1:]
    label = label.flatten()

    return label

def make_dataset(header, data):
    header_r = header[1:]
    data_r = data[:,1:]

    column = set()
    for i in range(len(data_r)):
        for j in range(len(header_r)):
            column.add(str(header_r[j] + '_' + str(data_r[i][j])))
    items = list(column)
    items.sort()

    dataset = []
    for i in range(len(data_r)):
        temp = []
        variable = []
        for j in range(len(header_r)):
            variable.append(header_r[j] +'_' + str(data_r[i][j]))
            # print(variable)
        for k in range(len(items)):
            if items[k] in variable:
                temp.append(1)
            else:
                temp.append(0)
        # print(temp)
        dataset.append(temp)
    dataset = np.array(dataset)

    return items, dataset
