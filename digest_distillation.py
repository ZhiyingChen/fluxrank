import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import read_data as rd
import change_qualification as cq

# 最后整理出来的 DataFrame 格式为
# digest | cmdb_id | kpi_name | change_start_time | po | pu

def digest_distillation(cmdb_features):
    """
    作用：
    对不同的cmdb进行分类，这里docker，os，db分开；用dbscan分类，距离指标为pearson
    
    参数：
    cmdb_features: 每个cmdb的特征(这里注意docker,os,db分开), np.array
    例：有2个cmdb：os_1, os_2，3个指标：kpi_1,kpi_2,kpi_3
    cmdb_features = [[o_11,u_11,o_12,u_12,o_13,u_13], [o_21,u_21,o_22,u_22,o_23,u_23]]
    其中o_ij：cmdb i中指标j的overflow概率；u_ij：cmdb i中指标j的underflow概率
    
    返回：
    digests: np.array, 每个cmdb所属的类别
    
    例：有9个cmdb，digests = [0,1,2,1,2,3,3,2,0]
    共4类，第i个cmdb的类别对应于digests[i]
    """
    clustering = DBSCAN(eps=0.07, min_samples=2, metric='precomputed')
    df_cmdb_features_T = pd.DataFrame(cmdb_features).T
    distance_matrix = df_cmdb_features_T.corr(method='pearson')
    distance_matrix = distance_matrix.fillna(0)
#    print(distance_matrix)
    digests = clustering.fit(abs(distance_matrix)).labels_
    k = max(digests)
    for i in range(len(digests)):
        if digests[i] == -1:
            digests[i] = k+1
            k+=1
    return np.array(digests)



if __name__=="__main__":
    # # 测试函数
    # cmdb_features =  np.random.random((10,10))
    # digests = digest_distillation(cmdb_features)
    # print(digests)
    
    # 提取初始数据
    filepath = 'db_oracle_11g.csv'
    original_data = pd.read_csv(filepath)
    
    
    # 故障时间
    fault_datetime = '2020-4-11 2:15:00'
    etimestamp = rd.datetime_timestamp(fault_datetime)
    
    # 整理数据
    # 提取所有网元和指标的信息
    cmdb_id_list = ['db_003', 'db_007']
    kpi_list = list(set(original_data['name']))
    
    cmdb_id_features_list = []
    for cmdb_id in cmdb_id_list:
        cmdb_id_features = []
        for kpi in kpi_list:
            df_cmdb_id_kpi = rd.get_cmbd_kpi_df(cmdb_id, kpi)
            times = np.array(df_cmdb_id_kpi['timestamp'])
            values = np.array(df_cmdb_id_kpi['value'])
            change_start_time_info = cq.get_change_start_timestamp(times, values)
            cmdb_id_features.extend(change_start_time_info[1:])
        cmdb_id_features_list.append(cmdb_id_features)
        
    digests = digest_distillation(np.array(cmdb_id_features_list))