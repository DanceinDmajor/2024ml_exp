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
np.random.seed(30)
shuffled_indices = np.random.permutation(len(df))
df = df.iloc[shuffled_indices].reset_index(drop=True)


# 定义计算基尼系数的函数
def gini(y):
    counter = Counter(y)
    total = len(y)
    return 1 - sum((count / total) ** 2 for count in counter.values())


# 定义CART算法决策树类
class DecisionTreeCART:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.features = X.columns
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or len(y) < self.min_samples_split or (
                self.max_depth is not None and depth >= self.max_depth):
            return y.mode()[0]

        best_feature, best_value, best_score, best_groups = None, None, float('inf'), None
        for feature in X.columns:
            values = X[feature].unique()
            for value in values:
                groups = self._split(X, y, feature, value)
                score = self._gini_index(groups, y)
                if score < best_score:
                    best_feature, best_value, best_score, best_groups = feature, value, score, groups

        if best_score == float('inf'):
            return y.mode()[0]

        left_tree = self._build_tree(best_groups[0][0], best_groups[0][1], depth + 1)
        right_tree = self._build_tree(best_groups[1][0], best_groups[1][1], depth + 1)
        return {'feature': best_feature, 'value': best_value, 'left': left_tree, 'right': right_tree}

    def _split(self, X, y, feature, value):
        left_mask = X[feature] <= value
        right_mask = X[feature] > value
        return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])

    def _gini_index(self, groups, classes):
        n_instances = float(sum([len(group[1]) for group in groups]))
        gini = 0.0
        for (X_group, y_group) in groups:
            size = len(y_group)
            if size == 0:
                continue
            score = sum((list(y_group).count(cls) / size) ** 2 for cls in classes)
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def predict(self, X):
        return X.apply(self._predict_row, axis=1)

    def _predict_row(self, row):
        node = self.tree
        while isinstance(node, dict):
            if row[node['feature']] <= node['value']:
                node = node['left']
            else:
                node = node['right']
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
model = DecisionTreeCART(max_depth=5, min_samples_split=4, min_samples_leaf=2)
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = np.mean(y_pred == y_test)
print("CART模型优化后的准确率:", accuracy)

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
