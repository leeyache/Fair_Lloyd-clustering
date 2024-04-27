import util.load
import numpy as np
import sys
import matplotlib.pyplot as plt
import util.load
from sklearn.decomposition import PCA
import ast
import util.dataprocess
import util.clustering_util
import Fair_Lloyd
import util.optimal_centers
def main():
    # 数据集所在文件夹
    config_root = 'Resource'
    # 选用的数据集
    config_name = sys.argv[1]
    # 加载配置文件
    config_parser = util.load.load_configuration(config_root,config_name)
    dataRoot = config_parser.get(config_name,'dataRoot')
    dataName = config_parser.get(config_name,'dataName')
    # 加载数据
    df = util.load.load_data(dataRoot,dataName)
    # 是否属于PCA
    PCAUse = config_parser[config_name].getboolean('PCA')
    # 获取聚类使用的数目
    k_number_str = config_parser[config_name].get('k_number')
    k_number_min,k_number_max = ast.literal_eval(k_number_str)
    k_number = range(k_number_min,k_number_max)
    #对数据进行缺省值处理
    util.dataprocess.fill_na(df)
    fair_column = config_parser[config_name].get('fair_column')
    # 获取聚类中不会使用的列
    drop_columns_str = config_parser[config_name].get('drop_columns')
    if drop_columns_str== '':
        drop_columns = None
    else:
        drop_columns = [column.strip() for column in drop_columns_str.split(',')]
    # 删除聚类中不需要的列
    df_for_encoder = util.dataprocess.drop_columns(df, drop_columns)
    # 对除了保护列之外的所有数据进行独热编码
    df_encode = util.dataprocess.one_hot(df_for_encoder,fair_column)
    df_for_scale = util.dataprocess.drop_columns(df_encode, fair_column)
    # 对数据进行标准化处理
    df_scale = util.dataprocess.scaler(df_for_scale)
    df_fair_column = df[fair_column]
    # 用于存储结果的列表
    result_list = [np.zeros(len(df_fair_column.unique())) for _ in k_number ]
    result_list = np.array(result_list)
    kmeans_result_list = [np.zeros(len(df_fair_column.unique())) for _ in k_number]
    kmeans_result_list = np.array(result_list)
    #寻找当前最优中心点使用的参数
    epsilon = config_parser[config_name].getfloat('epsilon')
    T = config_parser[config_name].getint('T')
    #迭代次数
    enum_number = int(sys.argv[2])
    filename_result = 'Result\\'+config_name+'_'+fair_column+'_' + str(enum_number)+'_'+str(k_number_min)+'_'+str(k_number_max)+'_result.txt'
    fig_name = 'Result\\'+config_name+'_'+fair_column + '_' + str(enum_number) + '_' + str(k_number_min) + '_' + str(
        k_number_max) + 'Lloyd_and_Fair_Lloyd_cost.png'
    Lloyd_fig = 'Result\\'+config_name+'_'+fair_column + '_' + str(enum_number) + '_' + str(k_number_min) + '_' + str(
        k_number_max) + 'Lloyd_cost.png'
    Fair_Lloyd_fig = 'Result\\'+config_name+'_'+fair_column + '_' + str(enum_number) + '_' + str(k_number_min) + '_' + str(
        k_number_max) + 'Fair_Lloyd_cost.png'

    if PCAUse:
        pca_number = config_parser[config_name].getint('PCA_number')
        pca = PCA(n_components=pca_number)
        data = pca.fit_transform(df_scale)
    else:
        data = df_scale
    for _ in range(enum_number):
        for cluster_number in k_number:

            assignment,centroids,group_index,name2ix,function_result,unique_values,M_eta,alpha_matrix,group_size_number_list,kmeans_assign,kmeans_centroids = Fair_Lloyd.Fair_Lloyd(data,cluster_number,df_fair_column,epsilon,T)
            result_list[cluster_number-k_number_min] += function_result
            cost_group = util.clustering_util.caculate_cost(cluster_number,data,centroids,group_index,assignment,name2ix,unique_values)
            kmeans_cost_group = util.clustering_util.caculate_cost(cluster_number,data,kmeans_centroids,group_index,kmeans_assign,name2ix,unique_values)
            kmeans_result_list[cluster_number-k_number_min] += kmeans_cost_group
            with open(filename_result, 'a', encoding='utf-8') as f:
                f.write("结果是：{}\n".format(function_result))
                f.write("映射为{}\n".format(name2ix))
                f.write("代价是{}\n".format(cost_group))
                f.write("kmeans代价是{}\n".format(kmeans_cost_group))
                f.flush()

    for i in range(len(result_list)):
        result_list[i] /= enum_number
        kmeans_result_list /= enum_number


    plt.figure()

    for group in unique_values:

        plt.plot(k_number, result_list[:,name2ix[group]], label=group)
        plt.plot(k_number, kmeans_result_list[:, name2ix[group]], label='kmeans__'+group)

    plt.xlabel('k')
    plt.ylabel('cost')
    plt.legend()
    plt.savefig(fig_name)

    plt.figure()
    for group in unique_values:
        plt.plot(k_number, result_list[:, name2ix[group]], label=group)
    plt.xlabel('k')
    plt.ylabel('cost')
    plt.legend()
    plt.savefig(Fair_Lloyd_fig)

    plt.figure()
    for group in unique_values:
        plt.plot(k_number, kmeans_result_list[:, name2ix[group]], label=group)
    plt.xlabel('k')
    plt.ylabel('cost')
    plt.legend()
    plt.savefig(Lloyd_fig)

    plt.show()

if __name__ == '__main__':
    main()

