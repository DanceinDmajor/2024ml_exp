import pandas as pd
import numpy as np
from collections import Counter
from math import log2
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('iris.csv')

# 将类别标签转为数值
df['Species'] = df['Species'].astype('category').cat.codes

# 手动打乱数据集
np.random.seed(42)
shuffled_indices = np.random.permutation(len(df))
df = df.iloc[shuffled_indices].reset_index(drop=True)

# 定义计算熵的函数
def entropy(y):
    counter = Counter(y)
    total = len(y)
    return -sum((count / total) * log2(count / total) for count in counter.values())

# 定义ID3算法决策树类
class DecisionTreeID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.features = X.columns
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1:
            return y.iloc[0]
        if len(y) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return y.mode()[0]

        best_feature = self._find_best_split(X, y)
        if best_feature is None:
            return y.mode()[0]

        tree = {best_feature: {}}
        for value in X[best_feature].unique():
            sub_X = X[X[best_feature] == value]
            sub_y = y[sub_X.index]
            tree[best_feature][value] = self._build_tree(sub_X, sub_y, depth + 1)

        return tree

    def _find_best_split(self, X, y):
        base_entropy = entropy(y)
        best_info_gain = 0
        best_feature = None

        for feature in X.columns:
            feature_values = X[feature].unique()
            feature_entropy = 0

            for value in feature_values:
                sub_y = y[X[feature] == value]
                feature_entropy += len(sub_y) / len(y) * entropy(sub_y)

            info_gain = base_entropy - feature_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature

        return best_feature

    def predict(self, X):
        return X.apply(self._predict_row, axis=1)

    def _predict_row(self, row):
        node = self.tree
        while isinstance(node, dict):
            feature = next(iter(node))
            node = node[feature].get(row[feature], y_train.mode()[0])  # 默认返回训练集中最常见的类别标签
        return node

# 手动划分训练集和测试集
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

X = df.drop(columns='Species')
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练决策树模型
model = DecisionTreeID3(max_depth=3)
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = np.mean(y_pred == y_test)
print("模型准确率:", accuracy)

# 输出测试集各样本的预测标签和真实标签
results = pd.DataFrame({'真实标签': y_test, '预测标签': y_pred})
print(results)

# 可视化预测结果
plt.figure(figsize=(12, 6))

# 散点图：真实标签 vs. 预测标签
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_test, palette='deep')
plt.title('真实标签')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_pred, palette='deep')
plt.title('预测标签')

plt.tight_layout()
plt.show()
