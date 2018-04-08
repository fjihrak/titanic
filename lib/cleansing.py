import numpy as np
import pandas as pd
import scipy.stats as scs

def remake_dataset(header, data, items):
    header_r = header[1:]
    data_r = data[:,1:]

    dataset = []
    for i in range(len(data)):
        temp = []
        variable = []
        for j in range(len(header)):
            variable.append(header[j] +'_' + str(data[i][j]))
            # print(variable)
        for k in range(len(items)):
            if items[k] in variable:
                temp.append(1)
            else:
                temp.append(0)
        # print(temp)
        dataset.append(temp)

    return dataset

def corr(data):
    st_corr = []
    for i in range(len(data.T)):
        for j in range(len(data.T)):
            corr = scs.pearsonr(data.T[i],data.T[j])[0]
            if i != j and (corr >=0.7 or corr <= -0.7) and set([i, j]) not in st_corr:
                st_corr.append(set([i, j]))

    return st_corr

def remove_columns(items, data, label, st_corr):
    remove_columns = []
    for i in range(len(st_corr)):
        column1 = list(st_corr[i])[0]
        column2 = list(st_corr[i])[1]
        corr1 = scs.pearsonr(data.T[column1],label)[0]
        corr2 = scs.pearsonr(data.T[column2],label)[0]
        # print(items[column1])
        # print(items[column2])
        # print(corr1)
        # print(corr2)
        if abs(corr1) > abs(corr2):
            remove_columns.append(items[column2])
        else:
            remove_columns.append(items[column1])
    return remove_columns

def cleansing(header, data, items, dataset, label):
    #変数間の強相関の検出
    c = corr(dataset)

    #強相関の除去
    rc = remove_columns(items, dataset, label, c)
    print('remove' + str(rc))
    items_r = items
    # print(remove_columns)
    # print(items)
    for i in range(len(rc)):
        items_r.remove(rc[i])
    # print(items_r)

    dataset_r = remake_dataset(header, data, items_r)

    #クレンジング済みモデリング用データセットのcsv書き出し
    # with open('dataset.csv', 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
    #     writer.writerow(items_r)
    #     writer.writerows(dataset_r)

    return items_r, dataset_r
