import pdb

import pandas as pd
import numpy as np

def kmeans_plusplus_initialization(data, k):
    # 从数据中随机选择一个样本作为第一个聚类中心
    centroids = [data[np.random.randint(len(data))]]

    # 选择剩余的聚类中心
    for _ in range(1, k):
        # 计算每个样本到最近聚类中心的距离的平方
        distances_sq = np.array([min([np.linalg.norm(x-c)**2 for c in centroids]) for x in data])
        # 根据距离的平方选择下一个聚类中心
        probabilities = distances_sq / distances_sq.sum()

        next_centroid_index = np.random.choice(len(data), p=probabilities)
        centroids.append(data[next_centroid_index])

    return np.array(centroids)


def assign_points_to_centers(data, centers):
    num_samples = data.shape[0]
    num_centers = centers.shape[0]

    # 初始化分配数组，用于存储每个样本点分配给的中心点索引
    assignments = np.zeros(num_samples, dtype=int)

    # 遍历每个样本点
    for i in range(num_samples):
        # 计算样本点与所有中心点之间的距离
        distances = np.linalg.norm(data[i] - centers, axis=1)

        # 找到距离最近的中心点的索引
        nearest_center_index = np.argmin(distances)

        # 将该样本点分配给最近的中心点
        assignments[i] = nearest_center_index


    return assignments

#名字到索引的映射
def name2index(unique_values):
    name2ix = {}
    for i, name in enumerate(unique_values):
        name2ix[name] = i
    return name2ix




# 计算每个种群的代价
def caculate_cost(k,data,centroids,group_index,assign,name2ix,unique_values):
    cost = np.zeros(len(unique_values))
    for group in unique_values:
        for i in range(k) :
            group_in_cur_clustering = group_index[name2ix[group]] & (assign == i)
            cluster_data = data[group_in_cur_clustering]
            distances = np.sum((cluster_data - centroids[i])**2,axis=1)

            cost[name2ix[group]] += sum(distances)
        cost[name2ix[group]] /= sum(group_index[name2ix[group]])
    return cost


