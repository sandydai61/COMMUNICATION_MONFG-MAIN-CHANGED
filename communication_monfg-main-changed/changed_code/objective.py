import numpy as np

""" Maximum_net_income 最大净收益
    Maximum overall performance 最大综合效能
    User satisfaction 用户满意度
"""    

"""
    计算目标函数值。
    #1、需要将参数和变量分开，参数确定的值写在函数的里面，保证写的函数没有问题
    #2、算法1和代码对应的地方确认
"""

def objective(p_e, p_h,P_grid, H_grid, P_w, P_pv,G_GT, G_GB, G_HR, d_e, d_h): #qu zhishuzhi
    """
    计算分布式能源站的净收益。

    参数：
    T: 时间周期数量
    p_e: 单位电价数组
    p_h: 单位热价数组
    p_e_prime: 从电网购得电价格数组
    p_h_prime: 从热网购得热价格数组
    c_k: 能量转换设备的成本系数数组
    f_k: 系统固定成本数组
    c_wp: 弃光惩罚系数数组
    P_grid: 分布式能源站发电量数组
    H_grid: 分布式能源站供热量数组
    H_k: 总供热量数组
    P_w: 弃风量数组
    P_e: 分布式能源站电需求量数组
    c_GT: 燃气轮机转换系数数组
    c_GB: 锅炉转换系数数组
    c_HR: 余热锅炉转换系数数组
    G_GT: 燃气轮机燃料成本数组
    G_GB: 锅炉燃料成本数组
    G_HR: 余热锅炉燃料成本数组

    返回：
    L_DES_k: 净收益值
    """
    #给参数赋值
    T = 24

    p_e_prime = np.array([6] * T)  # 从电网购得电价格数组  ?
    p_h_prime = np.array([7] * T)  # 从热网购得热价格数组    ?
    c_k = np.array([18] * T)  # 能量转换设备的成本系数数组
    f_k = np.array([10] * T)  # 系统固定成本数组
    c_wp = np.array([0.25] * T)  # 弃光惩罚系数数组  
    c_GT = np.array([0.34] * T)  # 燃气轮机转换系数数组
    c_GB = np.array([0.28] * T)  # 锅炉转换系数数组
    c_HR = np.array([0.4] * T)  # 余热锅炉转换系数数组

    eta_GT_e = 0.35  # 燃气轮机发电效率
    eta_ST_e = 0.42  # 蒸汽轮机发电效率
    eta_GT_h = 0.5  # 燃气轮机发电效率（热）
    eta_GB_h = 0.7  # 燃气锅炉发电效率
    eta_HR_h = 0.56  # 余热锅炉蒸汽轮机发热效率
    eta_ST_h = 0.38  # 蒸汽轮机发热效率
    eta_GB_s = 0.1  # 燃气锅炉余热传递效率
    eta_HR_s = 0.18  # 燃气锅炉余热传递效率

    P_e = (eta_GT_e * G_GT + 
             eta_ST_e * eta_HR_s * (eta_GB_s * G_GB + eta_GT_h * G_GT) + 
             P_w + P_pv + P_grid)
    
    # 计算热功率产量
    H_h = (eta_GB_h * G_GB + 
             eta_HR_h * (eta_HR_s * G_GB + eta_GT_h * G_GT) + 
             eta_ST_h * eta_HR_s * (eta_GB_s * G_GB + eta_GT_h * G_GT) + 
             H_grid)
    
    # 计算弃风量 P_curtail
    P_curtail = np.zeros(T)
    for k in range(T):
        if P_w[k] + P_grid[k] <= P_e[k]:
            P_curtail[k] = 0
        else:
            P_curtail[k] = P_w[k] + P_grid[k] - P_e[k]
    
    # 计算 f1
    f1 = np.zeros(T)
    for k in range(T):
        f1[k] = (p_e[k] * (P_e[k] + P_grid[k]) + p_h[k] * (H_h[k] + H_grid[k])
                 - (c_k[k] * (G_GT[k] + G_GB[k]) + f_k[k] + c_wp[k] * P_curtail[k])
                 - p_e_prime[k] * P_grid[k] - p_h_prime[k] * H_grid[k])
    
    # 计算 f2
    f2 = np.zeros(T)
    for k in range(T):
        f2[k] = (c_GT[k] * G_GT[k] + c_GB[k] * G_GB[k] + c_HR[k] * G_HR[k])
    
    # 计算净收益 L_DES_k
    L_DES_k = f1 - f2

    """
    计算用户获取最大综合效能的目标函数值。

    参数：
    T: 时间周期数量
    b_e: 电能需求系数
    a_e: 电能需求意愿系数
    d_e: 电能需求量
    b_h: 热能需求系数
    a_h: 热能需求意愿系数
    d_h: 热能需求量
    p_e: 电价格
    p_h: 热价格

    返回：
    W_n: 目标函数值
    """
    #T = 24  # 时间周期数量

    f3 = np.zeros(T)
    for n in range(T):
        f3[n] = (b_e[n] * d_e[n] - (a_e[n] / 2) * (d_e[n] ** 2)) + (b_h[n] * d_h[n] - (a_h[n] / 2) * (d_h[n] ** 2))

    W_n = f3 - (p_e * d_e + p_h * d_h)

    """
    计算用户满意度的目标函数值。
    
    参数：
    P_e_load: 不参加博弈时用户电负荷量 (24小时)
    P_h_load: 不参加博弈时用户热负荷量 (24小时)
    d_e: 用户电需求量 (24小时)
    d_h: 用户热需求量 (24小时)
    
    返回：
    W_m: 用户满意度函数值
    """
    
    numerator = 0
    denominator = 0
    
    for t in range(24):
        numerator += abs(P_e_load[t] - d_e[t]) + abs(P_h_load[t] - d_h[t])
        denominator += P_e_load[t] + P_h_load[t]
    
    W_m = 1 - numerator / denominator
    
    return [L_DES_k,W_n,W_m]

     
# 示例调用
T = 24  # 时间周期数量

# 定义各时间周期的参数
#p_e = np.array([10] * T)  # 单位电价数组
#p_h = np.array([8] * T)  # 单位热价数组  (出现在约束函数里面)
P_grid = np.array([20] * T)  # 分布式能源站发电量数组
H_grid = np.array([15] * T)  # 分布式能源站供热量数组
H_k = np.array([10] * T)  # 总供热量数组
P_w = np.array([12] * T)  # 弃风量数组
P_e = np.array([18] * T)  # 分布式能源站电需求量数组
P_pv = np.array([10] * T)
G_GT = np.array([1] * T)  # 燃气轮机燃料成本数组
G_GB = np.array([1] * T)  # 锅炉燃料成本数组
G_HR = np.array([1] * T)  # 余热锅炉燃料成本数组

# 定义各时间周期的参数
b_e = np.array([10] * T)  # 电能需求系数
a_e = np.array([2] * T)  # 电能需求意愿系数
d_e = np.array([5] * T)  # 电能需求量
b_h = np.array([8] * T)  # 热能需求系数
a_h = np.array([1.5] * T)  # 热能需求意愿系数
d_h = np.array([4] * T)  # 热能需求量
p_e = np.array([6] * T)  # 电价格 ?
p_h = np.array([7] * T)  # 热价格 ?

P_e_load = np.array([100] * T)  # 不参加博弈时用户电负荷量 (示例数据)
P_h_load = np.array([80] * T)  # 不参加博弈时用户热负荷量 (示例数据)
#d_e = np.array([95] * 24)  # 用户电需求量 (示例数据) ?
#d_h = np.array([75] * 24)  # 用户热需求量 (示例数据) ?

a_e = [8,10,10,10,10,8,8,8,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,8]
b_e = [80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,120,120,120,120,120,80]
a_h = [5,5,5,5,5,5,5,5,6,6,6,6,6,10,10,10,10,10,10,10,6,6,6,6]
b_h = [100,100,100,100,100,100,100,100,90,90,90,90,90,120,120,120,120,120,120,120,90,90,90,90]

[L_DES_k,W_n,W_m] = objective(p_e, p_h,P_grid, H_grid, P_w, P_pv,G_GT, G_GB, G_HR, d_e, d_h)
print("净收益值：", L_DES_k)
print("用户获取最大综合效能目标函数值：", W_n)
print("用户满意度目标函数值：", W_m)


def calculate_net_profit(p_e, p_h,P_grid, H_grid, P_w, P_pv,G_GT, G_GB, G_HR):
    """
    计算分布式能源站的净收益。

    参数：
    T: 时间周期数量
    p_e: 单位电价数组
    p_h: 单位热价数组
    p_e_prime: 从电网购得电价格数组
    p_h_prime: 从热网购得热价格数组
    c_k: 能量转换设备的成本系数数组
    f_k: 系统固定成本数组
    c_wp: 弃光惩罚系数数组
    P_grid: 分布式能源站发电量数组
    H_grid: 分布式能源站供热量数组
    H_k: 总供热量数组
    P_w: 弃风量数组
    P_e: 分布式能源站电需求量数组
    c_GT: 燃气轮机转换系数数组
    c_GB: 锅炉转换系数数组
    c_HR: 余热锅炉转换系数数组
    G_GT: 燃气轮机燃料成本数组
    G_GB: 锅炉燃料成本数组
    G_HR: 余热锅炉燃料成本数组

    返回：
    L_DES_k: 净收益值
    """
    #给参数赋值
    T = 10

    p_e_prime = np.array([6] * T)  # 从电网购得电价格数组  ?
    p_h_prime = np.array([7] * T)  # 从热网购得热价格数组    ?
    c_k = np.array([18] * T)  # 能量转换设备的成本系数数组
    f_k = np.array([10] * T)  # 系统固定成本数组
    c_wp = np.array([0.25] * T)  # 弃光惩罚系数数组  
    c_GT = np.array([0.34] * T)  # 燃气轮机转换系数数组
    c_GB = np.array([0.28] * T)  # 锅炉转换系数数组
    c_HR = np.array([0.4] * T)  # 余热锅炉转换系数数组

    eta_GT_e = 0.35  # 燃气轮机发电效率
    eta_ST_e = 0.42  # 蒸汽轮机发电效率
    eta_GT_h = 0.5  # 燃气轮机发电效率（热）
    eta_GB_h = 0.7  # 燃气锅炉发电效率
    eta_HR_h = 0.56  # 余热锅炉蒸汽轮机发热效率
    eta_ST_h = 0.38  # 蒸汽轮机发热效率
    eta_GB_s = 0.1  # 燃气锅炉余热传递效率
    eta_HR_s = 0.18  # 燃气锅炉余热传递效率

    P_e = (eta_GT_e * G_GT + 
             eta_ST_e * eta_HR_s * (eta_GB_s * G_GB + eta_GT_h * G_GT) + 
             P_w + P_pv + P_grid)
    
    # 计算热功率产量
    H_h = (eta_GB_h * G_GB + 
             eta_HR_h * (eta_HR_s * G_GB + eta_GT_h * G_GT) + 
             eta_ST_h * eta_HR_s * (eta_GB_s * G_GB + eta_GT_h * G_GT) + 
             H_grid)
    
    # 计算弃风量 P_curtail
    P_curtail = np.zeros(T)
    for k in range(T):
        if P_w[k] + P_grid[k] <= P_e[k]:
            P_curtail[k] = 0
        else:
            P_curtail[k] = P_w[k] + P_grid[k] - P_e[k]
    
    # 计算 f1
    f1 = np.zeros(T)
    for k in range(T):
        f1[k] = (p_e[k] * (P_e[k] + P_grid[k]) + p_h[k] * (H_h[k] + H_grid[k])
                 - (c_k[k] * (G_GT[k] + G_GB[k]) + f_k[k] + c_wp[k] * P_curtail[k])
                 - p_e_prime[k] * P_grid[k] - p_h_prime[k] * H_grid[k])
    
    # 计算 f2
    f2 = np.zeros(T)
    for k in range(T):
        f2[k] = (c_GT[k] * G_GT[k] + c_GB[k] * G_GB[k] + c_HR[k] * G_HR[k])
    
    # 计算净收益 L_DES_k
    L_DES_k = f1 - f2
    
    return L_DES_k

'''
# 示例调用
T = 10  # 时间周期数量

# 定义各时间周期的参数
p_e = np.array([10] * T)  # 单位电价数组
p_h = np.array([8] * T)  # 单位热价数组  (出现在约束函数里面)
P_grid = np.array([20] * T)  # 分布式能源站发电量数组
H_grid = np.array([15] * T)  # 分布式能源站供热量数组
H_k = np.array([10] * T)  # 总供热量数组
P_w = np.array([12] * T)  # 弃风量数组
P_e = np.array([18] * T)  # 分布式能源站电需求量数组
P_pv = np.array([10] * T)
G_GT = np.array([1] * T)  # 燃气轮机燃料成本数组
G_GB = np.array([1] * T)  # 锅炉燃料成本数组
G_HR = np.array([1] * T)  # 余热锅炉燃料成本数组

L_DES_k = calculate_net_profit(p_e, p_h,P_grid, H_grid, P_w, P_pv,G_GT, G_GB, G_HR)

#print("净收益值：", L_DES_k)
'''

def calculate_efficiency(b_e, a_e, d_e, b_h, a_h, d_h, p_e, p_h):
    """
    计算用户获取最大综合效能的目标函数值。

    参数：
    T: 时间周期数量
    b_e: 电能需求系数
    a_e: 电能需求意愿系数
    d_e: 电能需求量
    b_h: 热能需求系数
    a_h: 热能需求意愿系数
    d_h: 热能需求量
    p_e: 电价格
    p_h: 热价格

    返回：
    W_n: 目标函数值
    """
    T = 10  # 时间周期数量

    f3 = np.zeros(T)
    for n in range(T):
        f3[n] = (b_e[n] * d_e[n] - (a_e[n] / 2) * (d_e[n] ** 2)) + (b_h[n] * d_h[n] - (a_h[n] / 2) * (d_h[n] ** 2))

    W_n = f3 - (p_e * d_e + p_h * d_h)

    return W_n

'''
# 示例调用
T = 10  # 时间周期数量

# 定义各时间周期的参数
b_e = np.array([10] * T)  # 电能需求系数
a_e = np.array([2] * T)  # 电能需求意愿系数
d_e = np.array([5] * T)  # 电能需求量
b_h = np.array([8] * T)  # 热能需求系数
a_h = np.array([1.5] * T)  # 热能需求意愿系数
d_h = np.array([4] * T)  # 热能需求量
p_e = np.array([6] * T)  # 电价格
p_h = np.array([7] * T)  # 热价格

W_n = calculate_efficiency(b_e, a_e, d_e, b_h, a_h, d_h, p_e, p_h)

#print("用户获取最大综合效能目标函数值：", W_n)
'''

def calculate_satisfaction(P_e_load, P_h_load, d_e, d_h):
    """
    计算用户满意度的目标函数值。
    
    参数：
    P_e_load: 不参加博弈时用户电负荷量 (24小时)
    P_h_load: 不参加博弈时用户热负荷量 (24小时)
    d_e: 用户电需求量 (24小时)
    d_h: 用户热需求量 (24小时)
    
    返回：
    W_m: 用户满意度函数值
    """
    
    numerator = 0
    denominator = 0
    
    for t in range(24):
        numerator += abs(P_e_load[t] - d_e[t]) + abs(P_h_load[t] - d_h[t])
        denominator += P_e_load[t] + P_h_load[t]
    
    W_m = 1 - numerator / denominator
    
    return W_m

'''
# 示例调用
P_e_load = np.array([100] * 24)  # 不参加博弈时用户电负荷量 (示例数据)
P_h_load = np.array([80] * 24)  # 不参加博弈时用户热负荷量 (示例数据)
d_e = np.array([95] * 24)  # 用户电需求量 (示例数据)
d_h = np.array([75] * 24)  # 用户热需求量 (示例数据)

W_m = calculate_satisfaction(P_e_load, P_h_load, d_e, d_h)

#print("用户满意度目标函数值：", W_m)
'''

def check_constraints(P_e, H_h, d_e, d_h, alpha_e, alpha_h):
    """
    检查约束条件是否满足。

    参数：
    P_e: 分布式能源站产生的电量数组
    H_h: 分布式能源站产生的热量数组
    d_e: 用户所需的电量数组
    d_h: 用户所需的热量数组
    alpha_e: 电传输网络损系数
    alpha_h: 热传输网络损系数

    返回：
    返回等式插值的绝对值
    """
    
    # 计算电量约束条件
    lhs_e = np.sum(P_e)
    rhs_e = (1 + alpha_e) * np.sum(d_e)
    
    # 计算热量约束条件
    lhs_h = np.sum(H_h)
    rhs_h = (1 + alpha_h) * np.sum(d_h)
    
    # 检查约束条件是否满足
    constraint_e = abs(lhs_e - rhs_e)   
    constraint_h = abs(lhs_h - rhs_h)
    #constraint_e = abs(lhs_e - rhs_e)
    #constraint_h = abs(lhs_h - rhs_h)   # <=  jingdu  10-6
    #return [constraint_e,constraint_h]
    return [constraint_e, constraint_h]

'''
# 示例调用
P_e = np.array([20, 20, 20, 20, 20])  # 分布式能源站产生的电量数组 (示例数据)
H_h = np.array([15, 15, 15, 15, 15])  # 分布式能源站产生的热量数组 (示例数据)
d_e = np.array([19, 19, 19, 19, 19])  # 用户所需的电量数组 (示例数据)
d_h = np.array([14, 14, 14, 14, 14])  # 用户所需的热量数组 (示例数据)
alpha_e = 0.1  # 电传输网络损系数 (示例数据)
alpha_h = 0.1  # 热传输网络损系数 (示例数据)

constraints_satisfied = check_constraints(P_e, H_h, d_e, d_h, alpha_e, alpha_h)

#print("约束条件满足：", constraints_satisfied)
'''

def check_inequality_constraints(p_e, p_h, d_e, d_h, G_GT, G_GB, G_HR, G_GT_max, G_GB_max, G_HR_max):
    """
    检查不等式约束条件，返回具体的不等式插值。

    参数：
    p_e: 电价
    p_h: 热价
    d_e: 用户电需求量数组
    d_h: 用户热需求量数组
    G_GT: 燃气轮机输出数组
    G_GB: 燃气锅炉输出数组
    G_HR: 余热锅炉输出数组
    G_GT_max: 燃气轮机输出上限
    G_GB_max: 燃气锅炉输出上限
    G_HR_max: 余热锅炉输出上限

    返回：
    constraints: 字典，包含每个不等式的插值
    """
    # 检查不等式约束条件
    constraints = {
        'p_e >= 0': p_e,
        'p_h >= 0': p_h,
        'd_e >= 0': [de for de in d_e],
        'd_h >= 0': [dh for dh in d_h],
        '0 <= G_GT <= G_GT_max': [(G, G_GT_max) for G in G_GT],
        '0 <= G_GB <= G_GB_max': [(G, G_GB_max) for G in G_GB],
        '0 <= G_HR <= G_HR_max': [(G, G_HR_max) for G in G_HR]
    }
    
    return constraints

'''
# 示例调用
p_e = 10  # 电价
p_h = 8   # 热价
d_e = [20, 20, 20, 20, 20]  # 用户电需求量数组
d_h = [15, 15, 15, 15, 15]  # 用户热需求量数组
G_GT = [30, 30, 30, 30, 30] # 燃气轮机输出数组
G_GB = [25, 25, 25, 25, 25] # 燃气锅炉输出数组
G_HR = [10, 10, 10, 10, 10] # 余热锅炉输出数组
G_GT_max = 50  # 燃气轮机输出上限
G_GB_max = 40  # 燃气锅炉输出上限
G_HR_max = 20  # 余热锅炉输出上限

constraints = check_inequality_constraints(p_e, p_h, d_e, d_h, G_GT, G_GB, G_HR, G_GT_max, G_GB_max, G_HR_max)
for key, value in constraints.items():
    print(f"{key}: {value}")
'''