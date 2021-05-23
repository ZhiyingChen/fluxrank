import pandas as pd
import numpy as np

# 根据以上模块整理出来的 DataFrame 格式为
# digest | cmdb_id | kpi_name | change_start_time | po | pu


# 对于digest d做特征提取
def choose_kpi_candidate_set(df_digest, etimestamp):
    """

    作用：选取出 digest d 中的 change_start_time(平均) < etimestamp 的 kpi candidate set
    
    参数：df_digest     DataFrame
         digest | cmdb_id | kpi_name | change_start_time | po | pu
         
    返回：kpi_candidate_set     list
        存放 candidate_kpi 的 name
    """
    kpi_list = list(set(df_digest['kpi_name']))
    kpi_candidate_set = []
    for kpi in kpi_list:
        df_digest_kpi = df_digest.loc[df_digest['kpi_name'] == kpi,:]
        Tc_kpi_mean = np.mean(df_digest_kpi['change_start_time'])
        if Tc_kpi_mean < etimestamp:
            kpi_candidate_set.append(kpi)
    candidate_ratio = len(kpi_candidate_set)/len(kpi_list)
    return kpi_candidate_set, candidate_ratio

def get_df_digest_candidate(df_digest, kpi_candidate_set):
    """
    作用：从 df_digest 中提取出在 kpi_candidate_set 中的 kpi
    
    参数： df_digest     DataFrame
          digest | cmdb_id | kpi_name | change_start_time | po | pu
         kpi_candidate_set     list
          存放 candidate_kpi 的 name
    
    返回：df_digest_candidate    DataFrame
         digest | cmdb_id | kpi_name | change_start_time | po | pu
    
    """
        
    df_digest_candidate = df_digest[df_digest['kpi_name'].isin(kpi_candidate_set)]
    return df_digest_candidate

def extract_digest_features(df_digest_candidate, ratio):
    """
    作用：
    用于提取每个 digest 的特征

    参数：
    df_digest_candidate    DataFrame
         digest | cmdb_id | kpi_name | change_start_time | po | pu
    ratio : 该 igest 的candidate占比

    返回：
    df_digest_kpi_feature : DataFrame
        digest | cmdb_id | kpi_name | Tc | std | p_max
        
    df_digest_feature : DataFrame
        digest | ratio | max_Tc | min_Tc | sum_Tc | mean_Tc |
        max_std | min_std | sum_std | mean_std |
        max_d | min_d | sum_d | mean_d

    """
    
    digest = df_digest_candidate.loc[0,'digest']
    
    df_digest_feature = pd.DataFrame([{'digest': digest, 'ratio': ratio}])
    df_digest_kpi_feature = pd.DataFrame()
    
    kpi_list = list(set(df_digest_candidate['kpi_name']))
    Tc_kpi_mean_list = []
    Tc_kpi_std_list = []
    p_max_list = []
    for kpi in kpi_list:
        df_digest_kpi = df_digest_candidate.loc[df_digest_candidate['kpi_name'] == kpi,:]
       
        Tc_kpi_mean = np.mean(df_digest_kpi['change_start_time'])
        Tc_kpi_mean_list.append(Tc_kpi_mean)
        
        Tc_kpi_std = np.std(df_digest_kpi['change_start_time'])
        Tc_kpi_std_list.append(Tc_kpi_std)
        
        po_kpi_mean = np.mean(df_digest_kpi['po'])
        pu_kpi_mean = np.mean(df_digest_kpi['pu'])
        p_max = np.max([po_kpi_mean, pu_kpi_mean])
        p_max_list.append(p_max)
        
        kpi_feature = {'digest':digest, 'kpi_name':kpi, 'ratio':ratio, 'Tc': Tc_kpi_mean, 'std': Tc_kpi_std, 'p_max':p_max}
        df_kpi_feature = pd.DataFrame([kpi_feature])
        df_digest_kpi_feature = df_digest_kpi_feature.append(df_kpi_feature)
    
    df_digest_kpi_feature = df_digest_kpi_feature.sort_values(by=['p_max'], ascending = False)
    
    df_digest_feature.loc['max_Tc'] = np.max(Tc_kpi_mean_list)
    df_digest_feature.loc['min_Tc'] = np.min(Tc_kpi_mean_list)
    df_digest_feature.loc['sum_Tc'] = np.sum(Tc_kpi_mean_list)
    df_digest_feature.loc['mean_Tc'] = np.mean(Tc_kpi_mean_list)
    
    df_digest_feature.loc['max_std'] = np.max(Tc_kpi_std_list)
    df_digest_feature.loc['min_std'] = np.min(Tc_kpi_std_list)
    df_digest_feature.loc['sum_std'] = np.sum(Tc_kpi_std_list)
    df_digest_feature.loc['mean_std'] = np.mean(Tc_kpi_std_list)
    
    df_digest_feature.loc['max_d'] = np.max(p_max_list)
    df_digest_feature.loc['min_d'] = np.min(p_max_list)
    df_digest_feature.loc['sum_d'] = np.sum(p_max_list)
    df_digest_feature.loc['mean_d'] = np.mean(p_max_list)
    
    return df_digest_kpi_feature, df_digest_feature
    
    
    