import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
import read_data as rd

def get_popu(times, values, change_start_time):
    """
    作用：用KDE方法计算某条时间序列change_start_time后overflow和underflow的概率
    参数：
    times: timestamp序列，array
    values: times对应的值序列，array
    change_start_time: 变化开始时间，timestamp，注意这里的change_start_time是绝对时间
    返回：
    po: overflow概率
    pu: underflow概率
    """
    # 找到times里面对应change_start_time的下标
    change_start_time_pos = np.argwhere(times == change_start_time)[0][0]
    
    # bandwidth
    width = 0.1
    # values标准化，[0,1]之间
    X_min = np.min(values)
    X_max = np.max(values)
    X_std = (values - X_min) / (X_max - X_min)
    X_train = X_std[:change_start_time_pos]
    X_test = X_std[change_start_time_pos:]
    train_value = X_train.reshape(-1, 1)

    # 建立KDE模型
    train_distribution = KernelDensity(kernel='gaussian', bandwidth=width).fit(train_value)

    # 计算概率值
    po = 0
    pu = 0
    for t in range(len(X_test)):
        # overflow 概率
        prob_o = quad(lambda x: np.exp(train_distribution.score_samples(np.array(x).reshape(-1, 1))), X_test[t], np.inf)[0]
        # underflow 概率
        prob_u = quad(lambda x: np.exp(train_distribution.score_samples(np.array(x).reshape(-1, 1))), -np.inf, X_test[t])[0]
        po += np.log(prob_o)
        pu += np.log(prob_u)
    po = -po / len(X_test)
    pu = -pu / len(X_test)

    return po, pu


def get_change_start_timestamp(times, values, etimestamp):
    """
    作用：
    针对给定的一段时间序列和错误时间点，一阶差分后用3-sigma法（这里取2.5-sigma）找到异常开始时间，
    对于用3-sigma无法确定异常发生时间的时间序列（异常不明显或根本没有），
    找其偏离均值最大的点作为异常开始时间，顺便返回KDE之后的overflow和underflow值
    参数：
    times: 时间序列的timestamp数列，np.array
    values: 时间序列的value数列，np.array
    etimestamp: 时间序列的error_timestamp，int
    返回：
    一个list：3个元素
    [change_start_time，int
     po: overflow概率
     pu: underflow概率]
    """
    if len(values)>1:
        # 一阶差分
        div_times = times[1:]
        div_values = values[1:] - values[:-1]
    
        # 3-sigma法则
        std_div_values = np.std(div_values)
        mean_div_values = np.mean(div_values)
        up_thres = mean_div_values + 2.5 * std_div_values
        
        adjusted_div_values = abs(div_values - mean_div_values)
        bool_array = adjusted_div_values > up_thres

        # 如果没有可能异常点，返回0
        if sum(bool_array) == 0 or std_div_values == 0:
            # 返回第一个最大值所在的位置下标
            pos_max = np.argmax(adjusted_div_values)
            change_start_time = div_times[pos_max]
        else:
            # 从可疑点中找距离error_timestamp最近的时间点(此处可能会有问题，如果最近的是change end time怎么办？？？)
            min_index = np.argmin(abs(div_times[bool_array] - etimestamp))
            change_start_time = div_times[bool_array][min_index]
        
        if len(div_times[div_times < change_start_time]) >= 2 and len(div_times[div_times > change_start_time]) >= 2:
            po, pu = get_popu(times, values, change_start_time)
            return [change_start_time, po, pu]
        
    return [0, 0, 0]



if __name__=='__main__':
    
    # 测试数据
    df_test = rd.get_cmbd_kpi_df('db_008', 'Logic_Read_Per_Sec')
    
    # 故障时间
    fault_datetime = '2020-4-11 2:15:00'
    
    
    # 测试get_change_start_timestamp函数
    times = np.array(df_test['timestamp'])
    values = np.array(df_test['value'])
    etimestamp = rd.datetime_timestamp(fault_datetime)
    change_start_time_info = get_change_start_timestamp(times, values, etimestamp)
    