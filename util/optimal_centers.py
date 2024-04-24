import numpy as np

# 寻找当前最优中心，这是算法的核心代码
def optimal_centers(k,M_eta,group_size_number_list,alpha_matrix,group_index,data,unique_values,cur_assign,name2ix,enumrate_number = 100,epsilon = 1e-6):
    # 初始化lambda_list
    lambda_list = [1/len(unique_values) for _ in range(len(unique_values))]
    # 初始化function_result
    function_result = np.zeros(len(unique_values),dtype=int)
    lambda_adjust = 1/len(unique_values)
    inspect_every = len(unique_values)-1
    delta_result = delta_function(data,M_eta,k,group_index,cur_assign,group_size_number_list,unique_values,name2ix)

    for e_number in range(enumrate_number):
        for _ in range(inspect_every):
            optimal_centroids = lambda2centroids(lambda_list,M_eta,alpha_matrix,k)
            centroids_minus_eta_result = centroids_minus_eta(k, M_eta, optimal_centroids, alpha_matrix, unique_values, name2ix)
            function_result = calculate_function(delta_result , centroids_minus_eta_result)
            max_index = function_result.argmax()
            min_index = find_min_index(function_result,lambda_list)
            if abs(function_result[max_index]-function_result[min_index])<epsilon:
                return np.array(optimal_centroids),function_result
            lambda_list[min_index] -= lambda_adjust * 2 ** (-e_number-1)
            lambda_list[max_index] += lambda_adjust * 2 ** (-e_number-1)
    return np.array(optimal_centroids),function_result
# 计算F(x)函数
def calculate_function(delta_result,centroids_minus_eta_result):
    function_result = delta_result+centroids_minus_eta_result
    return function_result


# lambda 到 centers 的转换
def lambda2centroids(lambda_list,M_eta,alpha_matrix,k):
    centroids = [np.zeros(len(M_eta[0])) for _ in range(k)]
    for i in range(k):
        # 初始化分母
        denominator = 0
        # 初始化分子
        numerator = 0
        for j in range(len(M_eta)):
            denominator += lambda_list[j]*alpha_matrix[j][i]
            numerator += lambda_list[j]*alpha_matrix[j][i]*M_eta[j][i]
        centroids[i] = numerator/denominator
    return centroids


def delta_function(data,M_eta,k,group_index,cur_assign,group_size_number_list,unique_values,name2ix):
    delta_result = np.zeros(len(unique_values))
    for group in unique_values:
        for i in range(k):
            group_in_cur_clustering = group_index[name2ix[group]] & (cur_assign == i)
            distance = data[group_in_cur_clustering] - M_eta[name2ix[group]][i]
            norms = np.sum(distance**2, axis=1)

            sum_distance = sum(norms)
            delta_result[name2ix[group]] += sum_distance

        delta_result[name2ix[group]] /= group_size_number_list[name2ix[group]]

    return delta_result


def centroids_minus_eta(k,M_eta,optimal_centroids,alpha_matrix,unique_values,name2ix):
    centroids_minus_eta_result = np.zeros(len(unique_values))

    for group in unique_values:
        for i in range(k):
            centroids_minus_eta_result[name2ix[group]] += alpha_matrix[name2ix[group]][i]*np.linalg.norm(optimal_centroids[i]-M_eta[name2ix[group]][i],ord=2)**2
    return centroids_minus_eta_result


def find_min_index(function_result,lambda_list):
    sorted_indices = np.argsort(function_result)
    for i in range(len(sorted_indices)):
        j = sorted_indices[i]
        if lambda_list[j] > 0:
            return j

