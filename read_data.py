# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:26:05 2021

@author: CZY
"""
import pandas as pd
#import numpy as np
import time

def datetime_timestamp(dt):
    """
    作用：标准时间转换成Unix时间戳
    参数：
    dt: 标准时间结构："%Y-%m-%d %H:%M:%S"
    返回：
    value: Unix时间戳（毫秒级别）
    """
    # dt为字符串
    # 中间过程，一般都需要将字符串转化为时间数组
    time.strptime(dt, '%Y-%m-%d %H:%M:%S')
    # # time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88,
    # tm_isdst=-1) 将"2012-03-28 06:53:40"转化为时间戳
    s = time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))
    return int(s) * 1000

# 提取初始数据
filepath = 'db_oracle_11g.csv'
original_data = pd.read_csv(filepath)

# 故障时间
fault_datetime = '2020-4-11 2:15:00'
fault_timestamp = datetime_timestamp(fault_datetime)

# 整理数据
# 提取所有网元和指标的信息
cmdb_id_list = list(set(original_data['cmdb_id']))
kpi_list = list(set(original_data['name']))


def get_cmbd_kpi_df(cmdb_id, kpi_name):
    cmdb_kpi_df = original_data.loc[(original_data['cmdb_id']==cmdb_id) & (original_data['name']==kpi_name),:]
    cmdb_kpi_df = cmdb_kpi_df.sort_values(by=['timestamp'])
    return cmdb_kpi_df