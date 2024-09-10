import numpy as np
import matplotlib.pyplot as plt

# 定义 GMM 的参数
weights = np.array([0.5, 0.5])  # 高斯分布的权重
means = np.array([0, 5])  # 每个高斯分布的均值
variances = np.array([1, 2])  # 每个高斯分布的方差

# 定义生成一维数据点的函数
def sample_from_1d_gmm(weights, means, variances, n_samples=100):
    n_components = len(weights)  # GMM 中分量的个数

    # 根据权重从各个高斯分布中选择一个分布
    selected_components = np.random.choice(n_components, size=n_samples, p=weights)
    
    samples = np.zeros(n_samples)  # 存储生成的样本
    
    for i, component in enumerate(selected_components):
        mean = means[component]  # 选择的高斯分布的均值
        var = variances[component]  # 选择的高斯分布的方差
        # 从该高斯分布中采样一个一维数据点
        samples[i] = np.random.normal(mean, np.sqrt(var))
    
    return samples

# 生成100个数据点
sampled_data = sample_from_1d_gmm(weights, means, variances, n_samples=100)

# 绘制生成的样本数据
plt.hist(sampled_data, bins=30, density=True, alpha=0.6, color='g')
plt.title("1D Data Points Generated from GMM")
plt.xlabel("Value")
plt.ylabel("Density")
plt.grid(True)
plt.show()