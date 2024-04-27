import numpy as np
import util.clustering_util
import util.optimal_centers
import pdb
from sklearn.cluster import KMeans


def Fair_Lloyd(data,k,df_fair_column,epsilon=1e-6,enumrate_number = 100):
    # 初始化中心点，这里使用的是k-means++
    centroids = util.clustering_util.kmeans_plusplus_initialization(data, k)

    optimal_centroids = centroids
    # 初始化最初的分配
    initial_assignments = util.clustering_util.assign_points_to_centers(data, centroids)
    cur_assign = np.zeros(data.shape[0], dtype=int)
    update_assign = initial_assignments
    unique_values = df_fair_column.unique()
    # 初始化alpha矩阵
    alpha_matrix = np.zeros((len(unique_values),k))
    #每一个种群的总数目
    group_size_number_list = np.zeros(len(unique_values),dtype=int)
    # 每个种群对应的坐标
    group_index = [0 for _ in range(len(unique_values))]
    #初始化M矩阵
    M_eta = [[np.zeros(data.shape[1]) for _ in range(k)] for _ in range(len(unique_values))]
    # 名字到索引的映射
    name2ix = util.clustering_util.name2index(unique_values)
    for group in unique_values:

        group_index[name2ix[group]] = df_fair_column == group
        group_size_number_list[name2ix[group]] = sum(group_index[name2ix[group]])

    kmeans = KMeans(n_clusters=k, init=centroids,n_init=1).fit(data)

    kmeans_assign = kmeans.labels_
    k_means_centroids = kmeans.cluster_centers_
    # 算法收敛判断条件
    while np.count_nonzero(cur_assign != update_assign) > 10:

        for group in unique_values:

            for i in range(k):
                # 计算种群在该聚类中的位置
                group_in_cur_clustering = group_index[name2ix[group]]&(update_assign == i)
                # 计算种群在k 个聚类中alpha值
                alpha_matrix[name2ix[group]][i] = sum(group_in_cur_clustering)/group_size_number_list[name2ix[group]]
                if np.any(alpha_matrix == 1):
                    return Fair_Lloyd(data,k,df_fair_column,epsilon,enumrate_number)
                # 计算种群内的平均点
                if not sum(group_in_cur_clustering) == 0:
                    M_eta[name2ix[group]][i] = np.mean(data[group_in_cur_clustering],axis=0)
                else:
                    M_eta[name2ix[group]][i] = optimal_centroids[i]

        optimal_centroids,function_result = util.optimal_centers.optimal_centers(k,M_eta,group_size_number_list,alpha_matrix,group_index,data,unique_values,update_assign,name2ix,enumrate_number,epsilon)
        cur_assign = update_assign
        update_assign = util.clustering_util.assign_points_to_centers(data, optimal_centroids)

    print("差别有{}个".format(sum(cur_assign != update_assign)))
    for group in unique_values:

        for i in range(k):
            # 计算种群在该聚类中的位置
            group_in_cur_clustering = group_index[name2ix[group]] & (update_assign == i)
            # 计算种群在k 个聚类中alpha值
            alpha_matrix[name2ix[group]][i] = sum(group_in_cur_clustering) / group_size_number_list[name2ix[group]]
            if np.any(alpha_matrix == 1):
                return Fair_Lloyd(data, k, df_fair_column, epsilon, enumrate_number)
            # 计算种群内的平均值
            M_eta[name2ix[group]][i] = np.mean(data[group_in_cur_clustering], axis=0)

    return update_assign,optimal_centroids,group_index,name2ix,function_result,unique_values,M_eta,alpha_matrix,group_size_number_list,kmeans_assign,k_means_centroids







