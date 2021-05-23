# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:33:18 2021

@author: CZY
"""
import read_data as rd
import change_qualification as cq
import digest_distillation as dd
import digest_ranking as dr

import pandas as pd
import numpy as np

from tqdm import tqdm

filepath = 'db_oracle_11g.csv'
original_data = pd.read_csv(filepath)
# 故障时间
fault_datetime = '2020-4-11 2:15:00'
etimestamp = rd.datetime_timestamp(fault_datetime)

cmdb_id_list = list(set(original_data['cmdb_id']))
kpi_list = list(set(original_data['name']))

# df_cmdb_kpi_info 记录每个cmdb_id下的kpi的基本信息
# cmdb_id | kpi_name | change_start_time | po | pu
df_cmdb_kpi_info = pd.DataFrame()
# cmdb_id_features_list 记录每个cmdb_id的[o1,u1,o2,u2,...]
# [[o1,u1,o2,u2,...],[o1,u1,o2,u2,...],...,[o1,u1,o2,u2,...]]
cmdb_id_features_list = []

cmdb_bar = tqdm(cmdb_id_list)
kpi_bar = tqdm(kpi_list)
for cmdb_id in cmdb_bar:
    cmdb_bar.set_description('dealing with %s' % cmdb_id)
    cmdb_id_features = []
    for kpi in kpi_bar:
        #kpi_bar.set_description('dealing with %s' % kpi)
        
        cmdb_kpi_dict = {'cmdb_id': cmdb_id, 'kpi_name': kpi}
        cmdb_kpi_df = rd.get_cmbd_kpi_df(cmdb_id, kpi)
        times = np.array(cmdb_kpi_df['timestamp'])
        values = np.array(cmdb_kpi_df['value'])
        change_start_time_info = cq.get_change_start_timestamp(times, values, etimestamp)
        cmdb_kpi_dict['change_start_time'] = change_start_time_info[0]
        cmdb_kpi_dict['po'] = change_start_time_info[1]
        cmdb_kpi_dict['pu'] = change_start_time_info[2]
        df_cmdb_kpi_info = df_cmdb_kpi_info.append(pd.DataFrame([cmdb_kpi_dict]))
        
        cmdb_id_features.append(change_start_time_info[1])
        cmdb_id_features.append(change_start_time_info[2])
    cmdb_id_features_list.append(cmdb_id_features)
    
# 区分digest
# update   df_cmdb_kpi_info  DataFrame
# digest | cmdb_id | kpi_name | change_start_time | po | pu
digests = dd.digest_distillation(np.array(cmdb_id_features_list))
cmdb_digest_dict = dict(zip(cmdb_id_list, digests))
cmdb_digest_df = pd.DataFrame(columns=['cmdb_id','digest'])
cmdb_digest_df['cmdb_id'] = cmdb_id_list
cmdb_digest_df['digest'] = digests
df_cmdb_kpi_info = pd.merge(df_cmdb_kpi_info, cmdb_digest_df, on=['cmdb_id'])

# 获取每个digest的特征(features)
digests = list(set(digests))
df_all_digest_kpi_feature = pd.DataFrame()
df_all_digest_feature = pd.DataFrame()
for digest in tqdm(digests):
    
    df_digest = df_cmdb_kpi_info[df_cmdb_kpi_info['digest'] == digest]
    kpi_candidate_set, candidate_ratio = dr.choose_kpi_candidate_set(df_digest, etimestamp)
    df_digest_candidate = dr.get_df_digest_candidate(df_digest, kpi_candidate_set)
    df_digest_kpi_feature, df_digest_feature = dr.extract_digest_features(df_digest_candidate, candidate_ratio)
    
    df_all_digest_kpi_feature = df_all_digest_kpi_feature.append(df_digest_candidate)
    df_all_digest_feature = df_all_digest_feature.append(df_digest_feature)
    
df_all_digest_kpi_feature.to_csv('./df_all_digest_kpi_feature', index = False)
df_all_digest_feature.to_csv('./df_all_digest_feature', index = False)