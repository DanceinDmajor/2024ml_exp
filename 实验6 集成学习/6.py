import pandas as pd
import numpy as np

# 加载数据集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 数据预处理函数
def preprocess_data(df, is_train=True):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # 处理类别特征
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

    if is_train:
        # 丢弃不必要的特征
        df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    else:
        # 丢弃不必要的特征，但保留 PassengerId
        df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

    return df

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data, is_train=False)

# 分离特征和标签
X_train = train_data.drop(columns=['Survived', 'PassengerId'])
y_train = train_data['Survived']
X_test = test_data.drop(columns=['PassengerId']).copy()

# 标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CARTDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(np.array(X), np.array(y))

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feat, best_thresh = self._best_criteria(X, y, n_features)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._gini(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _most_common_label(self, y):
        if len(y) == 0:
            return None
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in np.array(X)])

    def _predict(self, inputs):
        node = self.tree
        while node.value is None:
            if inputs[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        y = np.array(y)  # 确保y是NumPy数组
        self.trees = []
        for _ in range(self.n_trees):
            tree = CARTDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]  # 直接返回y的索引

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return self._majority_vote(tree_preds)

    def _majority_vote(self, predictions):
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

class SVM:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = self.compute_kernel(X, X)

        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))
        A = y.values.reshape(1, -1).astype(float)
        b = np.zeros(1)

        # Solve the quadratic programming problem
        self.alpha = self.solve_qp(P, q, G, h, A, b)

        sv = self.alpha > 1e-5
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]
        self.alpha = self.alpha[sv]

    def compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'polynomial':
            return (np.dot(X1, X2.T) + 1) ** 2
        elif self.kernel == 'rbf':
            gamma = 0.1
            K = np.exp(-gamma * (np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2))
            return K
        else:
            raise ValueError("Unknown kernel function")

    def solve_qp(self, P, q, G, h, A, b):
        from cvxopt import matrix, solvers
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
        sol = solvers.qp(P, q, G, h, A, b)
        return np.ravel(sol['x'])

    def predict(self, X):
        K = self.compute_kernel(X, self.support_vectors)
        return np.sign(np.dot(K, self.alpha * self.support_vector_labels))


class KNN:
    def __init__(self, k=3, distance='euclidean', weights='uniform'):
        self.k = k
        self.distance = distance
        self.weights = weights
        self.S_inv = None  # Inverse of covariance matrix for Mahalanobis distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)  # 将 y 转换为 NumPy 数组
        if self.distance == 'mahalanobis':
            cov_matrix = np.cov(X, rowvar=False)
            self.S_inv = np.linalg.inv(cov_matrix)

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        if self.distance == 'euclidean':
            distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        elif self.distance == 'mahalanobis':
            distances = [np.sqrt((x - x_train).T.dot(self.S_inv).dot(x - x_train)) for x_train in self.X_train]
        elif self.distance == 'manhattan':
            distances = [np.sum(np.abs(x - x_train)) for x_train in self.X_train]
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if self.weights == 'uniform':
            most_common = np.bincount(k_nearest_labels).argmax()
        elif self.weights == 'distance':
            weights = 1 / (np.array(distances)[k_indices] + 1e-5)
            most_common = np.bincount(k_nearest_labels, weights=weights).argmax()
        else:
            raise ValueError(f"Unknown weights option: {self.weights}")

        return most_common


def k_fold_cross_validation(X, y, k_values, distance='euclidean', weights='uniform', n_splits=5):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_size = n_samples // n_splits

    best_k = None
    best_score = 0

    for k in k_values:
        scores = []
        for i in range(n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i != n_splits - 1 else n_samples
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate((indices[:val_start], indices[val_end:]))

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            knn = KNN(k=k, distance=distance, weights=weights)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            score = np.mean(y_pred == y_val)
            scores.append(score)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    return best_k, best_score


class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, activation='relu', use_dropout=False, dropout_rate=0.5, use_l2_regularization=False, l2_lambda=0.01):
        self.weights_input_hidden = np.random.rand(n_inputs, n_hidden)
        self.weights_hidden_output = np.random.rand(n_hidden, n_outputs)
        self.activation = activation
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_l2_regularization = use_l2_regularization
        self.l2_lambda = l2_lambda

    def _activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def _activate_derivative(self, x):
        if self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'sigmoid':
            return x * (1 - x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def fit(self, X, y, epochs=1000, learning_rate=0.1):
        y = y.values.reshape(-1, 1)
        for epoch in range(epochs):
            # 前向传播
            hidden_input = np.dot(X, self.weights_input_hidden)
            hidden_output = self._activate(hidden_input)
            if self.use_dropout:
                dropout_mask = (np.random.rand(*hidden_output.shape) > self.dropout_rate).astype(float)
                hidden_output *= dropout_mask / (1.0 - self.dropout_rate)
            final_input = np.dot(hidden_output, self.weights_hidden_output)
            final_output = self._activate(final_input)

            # 计算误差
            error = y - final_output
            final_output_delta = error * self._activate_derivative(final_output)
            hidden_error = final_output_delta.dot(self.weights_hidden_output.T)
            hidden_output_delta = hidden_error * self._activate_derivative(hidden_output)

            # 更新权重
            if self.use_l2_regularization:
                self.weights_hidden_output += hidden_output.T.dot(final_output_delta) * learning_rate - self.l2_lambda * self.weights_hidden_output
                self.weights_input_hidden += X.T.dot(hidden_output_delta) * learning_rate - self.l2_lambda * self.weights_input_hidden
            else:
                self.weights_hidden_output += hidden_output.T.dot(final_output_delta) * learning_rate
                self.weights_input_hidden += X.T.dot(hidden_output_delta) * learning_rate

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden)
        hidden_output = self._activate(hidden_input)
        final_input = np.dot(hidden_output, self.weights_hidden_output)
        final_output = self._activate(final_input)
        return (final_output > 0.5).astype(int).flatten()




class VotingClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        for clf in self.classifiers:
            clf.fit(X, y)

    def predict(self, X):
        predictions = np.array([clf.predict(X).flatten() for clf in self.classifiers])
        # 将 SVM 的预测结果从 -1 和 1 转换为 0 和 1
        predictions = np.where(predictions == -1, 0, predictions)
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_vote


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier


k_values = range(1, 21)

# 固定 distance 和 weights 参数
distance = 'euclidean'
weights = 'uniform'

best_k, best_score = k_fold_cross_validation(X_train, y_train, k_values, distance=distance, weights=weights)

print(f"Best k: {best_k} with cross-validation score: {best_score}")

# 使用最佳 k 值来训练 KNN 模型
optimized_knn = KNN(k=best_k, distance=distance, weights=weights)
optimized_knn.fit(X_train, y_train)


# 基学习器
classifiers = [
    ('RandomForest', RandomForest(n_trees=100, max_depth=5)),
    ('SVM', SVM()),
    ('KNN', optimized_knn),
    ('NN_sigmoid', NeuralNetwork(n_inputs=X_train.shape[1], n_hidden=10, n_outputs=1, activation='sigmoid', use_dropout=True, dropout_rate=0.5, use_l2_regularization=True, l2_lambda=0.01))
]

# 分割数据集
kf = KFold(n_splits=5, shuffle=True, random_state=42)
S_train = np.zeros((X_train.shape[0], len(classifiers)))
S_test = np.zeros((X_test.shape[0], len(classifiers)))

for i, (name, clf) in enumerate(classifiers):
    S_test_i = np.zeros((X_test.shape[0], kf.n_splits))
    for j, (train_index, val_index) in enumerate(kf.split(X_train)):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        clf.fit(X_train_fold, y_train_fold)
        S_train[val_index, i] = clf.predict(X_val_fold)
        S_test_i[:, j] = clf.predict(X_test)

    S_test[:, i] = S_test_i.mean(axis=1)

# 将预测结果作为新特征
stacked_features_train = pd.DataFrame(S_train, columns=[name for name, _ in classifiers])
stacked_features_test = pd.DataFrame(S_test, columns=[name for name, _ in classifiers])

# 训练元学习器
meta_clf = LogisticRegression()
meta_clf.fit(stacked_features_train, y_train)

# 使用交叉验证评估堆叠模型性能
scores = []
for train_idx, val_idx in kf.split(X_train):
    X_train_fold, X_val_fold = stacked_features_train.iloc[train_idx], stacked_features_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    meta_clf.fit(X_train_fold, y_train_fold)
    y_pred = meta_clf.predict(X_val_fold)
    score = np.mean(y_pred == y_val_fold)
    scores.append(score)

print(f"Stacked model cross-validation accuracy: {np.mean(scores):.4f}")

# 在测试集上进行预测
y_pred = meta_clf.predict(stacked_features_test)

# 提交结果
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": y_pred
})

submission.to_csv('submission.csv', index=False)
