# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

from clean_data import process_text


def plot_text_length(filename):
    # 文本长度的范围
    df = pd.read_csv(filename)
    texts = df['微博中文内容']
    texts.fillna("无", inplace=True)

    # 获取所有文本的长度
    all_length = []
    for content in texts:
        try:
            all_length.append(len(content))
        except Exception:
            print(content)

    plt.hist(all_length, bins=30) #画条形图
    plt.show()
    print(np.mean(np.array(all_length) < 170))
    df[:10].to_csv("./data/train.csv", index=False)


# 可视化语料序列长度, 可见文本的长度都在160以下
# plot_text_length('./data/train0.csv')
# plot_text_length('./data/test0.csv')
random.seed(20)

def cut_fold(k):
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")
    train_df['微博中文内容'].fillna('无', inplace=True)
    test_df['微博中文内容'].fillna('无', inplace=True)
    train_df = train_df.loc[train_df["情感倾向"].isin(['-1', '0', '1'])] #筛选出 情感倾向在-1 0 1 之间的
    # 数据清洗
    # train_df = process_text(train_df)
    # test_df = process_text(test_df)

    index = set(range(train_df.shape[0])) #读取第0维的长度
    #set   是一个不允许内容重复的组合，而且set里的内容位置是随意的，所以不能用索引列出。
    train_fold = []
    dev_fold = []
    for i in range(k):
        dev = random.sample(index, 2000)  #从index中随机选择2000个返回
        train = index - set(dev)
        print("Dev Number:", len(dev))
        print("Train Number:", len(train))
        dev_fold.append(dev)
        train_fold.append(train)

    for i in range(k):
        print("Fold", i)
        path = "./data/data_" + str(i)
        if not os.path.exists(path):
            os.makedirs(path)
        dev_index = list(dev_fold[i])
        train_index = list(train_fold[i])
        train_df.iloc[train_index].to_csv("./data/data_{}/train.csv".format(i), index=False)
        #iloc 根据train_Index确定位置
        train_df.iloc[dev_index].to_csv("./data/data_{}/dev.csv".format(i), index=False)
        test_df.to_csv("./data/data_{}/test.csv".format(i), index=False)


cut_fold(5)
