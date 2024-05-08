import pandas as pd
import numpy as np

# 读取 Excel 文件
data = pd.read_excel('exp2_2.xlsx', header=[0, 1])

# 分割数据为类别1和类别2
class1_data = data['类1']
class2_data = data['类2']

# 计算类别1中每个特征的均值和方差
class1_mean = class1_data.mean()
class1_variance = class1_data.var()

# 计算类别2中每个特征的均值和方差
class2_mean = class2_data.mean()
class2_variance = class2_data.var()

# 打印结果
print("类别1 中的均值和方差：")
for feature in class1_data.columns:
    print(f"特征 {feature} 的均值：\n", class1_mean[feature])
    print(f"特征 {feature} 的方差：\n", class1_variance[feature])

print("\n类别2 中的均值和方差：")
for feature in class2_data.columns:
    print(f"特征 {feature} 的均值：\n", class2_mean[feature])
    print(f"特征 {feature} 的方差：\n", class2_variance[feature])
# 定义函数来计算均值和协方差矩阵
def estimate_mean_covariance(data):
    mean = data.mean()
    covariance = data.cov()
    return mean, covariance

# 定义函数来计算任意两个特征的组合的均值和协方差矩阵
def compute_feature_combinations(data):
    for i, feature1 in enumerate(data.columns):
        for j, feature2 in enumerate(data.columns):
            if i < j:  # 避免重复计算
                combined_data = data[[feature1, feature2]]
                mean, covariance = estimate_mean_covariance(combined_data)
                print(f"特征组合 ({feature1}, {feature2}):")
                print("均值：\n", mean)
                print("协方差矩阵：\n", covariance)
                print()

# 计算类别1中任意两个特征的组合的均值和协方差矩阵
print("类别1 中任意两个特征的组合的均值和协方差矩阵：")
compute_feature_combinations(class1_data)

# 计算类别2中任意两个特征的组合的均值和协方差矩阵
print("类别2 中任意两个特征的组合的均值和协方差矩阵：")
compute_feature_combinations(class2_data)
# 计算类别1中三个特征的均值和协方差矩阵
class1_mean, class1_covariance = estimate_mean_covariance(class1_data)
print("类别1 中三个特征的均值：\n", class1_mean)
print("\n类别1 中三个特征的协方差矩阵：\n", class1_covariance)

# 计算类别2中三个特征的均值和协方差矩阵
class2_mean, class2_covariance = estimate_mean_covariance(class2_data)
print("\n类别2 中三个特征的均值：\n", class2_mean)
print("\n类别2 中三个特征的协方差矩阵：\n", class2_covariance)

# 定义函数来计算均值和对角协方差矩阵的参数
def estimate_mean_diagonal_covariance_params(data):
    mean = data.mean()
    covariance_diag = data.var()  # Variance along each dimension
    return mean, covariance_diag

# 计算类别1中的均值和对角协方差矩阵的参数
class1_mean, class1_covariance_diag = estimate_mean_diagonal_covariance_params(class1_data)
print("类别1 中的均值：\n", class1_mean)
print("\n类别1 中的对角协方差矩阵的参数：\n", class1_covariance_diag)

# 计算类别2中的均值和对角协方差矩阵的参数
class2_mean, class2_covariance_diag = estimate_mean_diagonal_covariance_params(class2_data)
print("\n类别2 中的均值：\n", class2_mean)
print("\n类别2 中的对角协方差矩阵的参数：\n", class2_covariance_diag)
