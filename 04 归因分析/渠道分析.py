import numpy as np # 导入NumPy
import pandas as pd # 导入Pandas
df_data = pd.read_csv('./渠道转化.csv') # 载入数据

df_data = df_data.sort_values(['用户cookie', '时戳'], ascending=[False, True]) # 按用户cookie 进行时戳排序

df_data['访问次序'] = df_data.groupby('用户cookie').cumcount() + 1 # 增加一个“访问次序”字段

df_paths = df_data.groupby('用户cookie')['渠道'].aggregate( # 根据用户cookie 构建路径（各渠道间的用户旅程）
    lambda x: x.unique().tolist()).reset_index() #lambda 后面是匿名函数
df_last_step = df_data.drop_duplicates('用户cookie', keep='last')[['用户cookie', '是否转化']] # 保留用户最后的一条记录（确定是否转化）
df_paths = pd.merge(df_paths, df_last_step, how='left', on='用户cookie') # 合并路径信息和用户是否转化的信息

df_paths['路径'] = np.where( df_paths['是否转化'] == 0, # 添加开始和转化结果
['开始, '] + df_paths['渠道'].apply(', '.join) + [', 未转化'], # 未购买游戏
['开始, '] + df_paths['渠道'].apply(', '.join) + [', 成功转化']) # 购买游戏
df_paths['路径'] = df_paths['路径'].str.split(', ') # 分割字符串，重新生成字符串列表
df_paths = df_paths[['用户cookie', '路径']] # 删除除“用户cookie”和“路径”之外的其他字段

path_list = df_paths['路径'] # 创建路径列表对象

total_conversions = sum(path.count('成功转化') for path in df_paths['路径'].tolist()) # 整体转化数
conversion_rate = total_conversions / len(path_list) # 基准转化率
print('整体转化数：',total_conversions) # 输出整体转化数
print('基准转化率：',conversion_rate) # 输出基准转化率

def transition_states(path_list): # 构建中间转换状态计数函数
    unique_channels = set(x for element in path_list for x in element) # 独立路径列表
    transition_states = {x + '>' + y: 0 for x in unique_channels for y in unique_channels} # 中间状态列表
    for possible_state in unique_channels: # 遍历所有独立路径
        if possible_state not in ['成功转化', '未转化']: # 最终转化步骤之前的所有状态
            for user_path in path_list: # 遍历路径列表
                if possible_state in user_path: # 如果可能状态在该路径中
                    indices = [i for i, s in enumerate(user_path) if possible_state in s] # 设定索引
                    for col in indices:
                        transition_states[user_path[col] + '>' + user_path[col + 1]] += 1 # 计数值加1
    return transition_states # 返回计数值


trans_states = transition_states(path_list) # 调用中间转换状态计数函数

from collections import defaultdict # 导入defaultdict 模块
def transition_prob(path_list, trans_dict): # 构建计算状态间过渡概率的函数
    unique_channels = set(x for element in path_list for x in element) # 独立路径列表
    trans_prob = defaultdict(dict) # 过渡概率
    for state in unique_channels: # 遍历所有独立路径
        if state not in ['成功转化', '未转化']: # 最终转化步骤之前的所有状态
            counter = 0 # 初始化counter
            index = [i for i, s in enumerate(trans_dict) if state + '>' in s] # 索引列表
            for col in index:
                if trans_dict[list(trans_dict)[col]] > 0:
                    counter += trans_dict[list(trans_dict)[col]] # 转化总计数值加1
                    for col in index:
                        if trans_dict[list(trans_dict)[col]] > 0:
                            state_prob = float((trans_dict[list(trans_dict)[col]])) / float(counter) # 计算过渡概率
                            trans_prob[list(trans_dict)[col]] = state_prob # 过渡概率结果
    return trans_prob # 返回过渡概率的列表


trans_prob = transition_prob(path_list, trans_states) # 调用计算状态间过渡概率的函数

def transition_matrix(path_list, transition_probabilities): # 构建过渡矩阵函数
    trans_matrix = pd.DataFrame() # 创建过渡矩阵对象
    unique_channels = set(x for element in path_list for x in element) # 独立渠道数
    for channel in unique_channels: # 遍历所有渠道
        trans_matrix[channel] = 0.00 # 初始化过渡矩阵对象
        trans_matrix.loc[channel] = 0.00 # 初始化过渡矩阵对象中的元素
        trans_matrix.loc[channel][channel] = 1.0 if channel in [' 成功转化', ' 未转化'] else 0.0 # 分别给元素赋默认值1 和0
    for key, value in transition_probabilities.items(): # 遍历所有可能的过渡状态
        origin, destination = key.split('>') # 用> 拆分元素
        trans_matrix.at[origin, destination] = value # 给元素赋值
    return trans_matrix # 返回过渡矩阵对象

trans_matrix = transition_matrix(path_list, trans_prob) # 调用过渡矩阵函数


def removal_effects(df, conversion_rate): # 计算移除效应系数的函数
    removal_effects_dict = {} # 初始化集合
    channels = [channel for channel in df.columns if channel not in ['开始','未转化','成功转化']] # 渠道列表
    for channel in channels: # 遍历每一个渠道
        removal_df = df.drop(channel, axis=1).drop(channel, axis=0) # 移除渠道
        for column in removal_df.columns: # 遍历每一列
            # 构建移除该渠道后的Dataframe 对象，即removal_df
            row_sum = np.sum(list(removal_df.loc[column]))
            null_pct = float(1) - row_sum
            if null_pct != 0:
                removal_df.loc[column]['未转化'] = null_pct
            removal_df.loc['未转化']['未转化'] = 1.0
        # 求移除该渠道之后的转化率
        removal_to_conv = removal_df[['未转化', '成功转化']].drop(['未转化', '成功转化'], axis=0) #其它渠道转化率
        removal_to_non_conv = removal_df.drop(['未转化', '成功转化'], axis=1).drop(['未转化', '成功转化'], axis=0) #其它渠道之间转化率
        removal_inv_diff = np.linalg.inv(np.identity(len(removal_to_non_conv.columns)) - np.asarray(removal_to_non_conv)) #逆矩阵
        removal_dot_prod = np.dot(removal_inv_diff, np.asarray(removal_to_conv)) # 渠道间转化率与最终转化率相乘
        removal_cvr = pd.DataFrame(removal_dot_prod, index=removal_to_conv.index)[[1]].loc['开始'].values[0] # 移除该渠道之后的转化率
        removal_effect = 1 - removal_cvr / conversion_rate # 求出该渠道的移除效应系数
        removal_effects_dict[channel] = removal_effect # 将结果赋给移除效应系数字典
    return removal_effects_dict # 返回移除效应系数字典

removal_effects_dict = removal_effects(trans_matrix, conversion_rate) # 调用计算移除效应系数的函数
 