import pandas as pd
import numpy as np

# 1. 加载数据集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 2. 数据预处理
# 2.1 处理缺失值
train_data.fillna(train_data.mean(numeric_only=True), inplace=True)
test_data.fillna(test_data.mean(numeric_only=True), inplace=True)
# 2.2 处理分类数据
def classifyOrdinary(dataframe: pd.DataFrame):
    # 对于'industry'列进行one-hot编码
    industry_dummies = pd.get_dummies(dataframe['industry'], prefix='industry')
    dataframe = pd.concat([dataframe, industry_dummies], axis=1)
    dataframe.drop('industry', axis=1, inplace=True)

    # 创建class特征 映射器，将A-1,B-2...F-6
    func = lambda x: ord(x) - 64  # ord()将字母转变为ASCII码
    # 将class特征分类
    dataframe['class'] = dataframe['class'].apply(func)

    # 创建编码映射器
    mapper = {
        'work_year': {
            '< 1 year': 0,
            '1 year': 1,
            '2 years': 2,
            '3 years': 3,
            '4 years': 4,
            '5 years': 5,
            '6 years': 6,
            '7 years': 7,
            '8 years': 8,
            '9 years': 9,
            '10+ years': 10,
        },
        'employer_type': {
            '普通企业': 1,
            '幼教与中小学校': 2,
            '政府机构': 3,
            '上市企业': 4,
            '高等教育机构': 5,
            '世界五百强': 6
        }
    }
    # 离散化
    dataframe.replace(mapper, inplace=True)
    return dataframe

# 2.3 日期处理格式
# 假设有日期字段 issue_date，将其转换为距离当前日期的天数
train_data['issue_date'] = pd.to_datetime(train_data['issue_date'])
test_data['issue_date'] = pd.to_datetime(test_data['issue_date'])
current_date = pd.to_datetime('now', utc=True).tz_localize(None)
train_data['issue_days_ago'] = (current_date - train_data['issue_date']).dt.days
test_data['issue_days_ago'] = (current_date - test_data['issue_date']).dt.days
# 删除原始日期字段
train_data.drop(columns=['issue_date'], inplace=True)
test_data.drop(columns=['issue_date'], inplace=True)

# 2.4 无关属性的删除
# 根据字段说明删除不需要的属性
drop_columns = ['loan_id', 'user_id']  # columns to be dropped

# Check if column exists before dropping
for column in drop_columns:
    if column in train_data.columns:
        train_data.drop(columns=[column], inplace=True)
    if column in test_data.columns:
        test_data.drop(columns=[column], inplace=True)
# 2.5 连续属性的离散化（如果有）
# 需要离散化的列
columns_to_discretize = ['total_loan', 'interest', 'monthly_payment', 'debt_loan_ratio', 'scoring_low', 'scoring_high',
                         'recircle_b', 'recircle_u', 'early_return_amount_3mon', 'early_return_amount', 'title']

# 对每个列进行等频分箱
for column in columns_to_discretize:
    # 计算分箱结果
    train_data[column + '_bins'] = pd.qcut(train_data[column], q=10, labels=False, duplicates='drop')
    test_data[column + '_bins'] = pd.qcut(test_data[column], q=10, labels=False, duplicates='drop')

    # 删除原始列
    train_data.drop(columns=[column], inplace=True)
    test_data.drop(columns=[column], inplace=True)

# 3. 实现贝叶斯分类器
class NaiveBayesClassifier:
    def __init__(self):
        self.priors = None
        self.likelihoods = None

    def fit(self, X, y):
        self.priors = {}
        self.likelihoods = {}

        # 计算先验概率
        self.priors[1] = np.sum(y) / len(y)
        self.priors[0] = 1 - self.priors[1]

        # 计算条件概率
        for col in X.columns:
            self.likelihoods[col] = {}
            for val in X[col].unique():
                self.likelihoods[col][val] = {}
                for label in [0, 1]:
                    self.likelihoods[col][val][label] = ((X[col] == val) & (y == label)).sum() / (y == label).sum()

    def predict(self, X):
        preds = []
        for idx, row in X.iterrows():
            probs = {label: np.log(self.priors[label]) for label in [0, 1]}
            for col, val in row.items():
                if val in self.likelihoods[col]:
                    for label in [0, 1]:
                        probs[label] += np.log(self.likelihoods[col][val][label] + 1e-6)  # Laplace smoothing
            preds.append(max(probs, key=probs.get))
        return preds

# 4. 训练模型并进行预测
X_train = train_data.drop(columns=['isDefault'])
y_train = train_data['isDefault']
X_test = test_data.drop(columns=['isDefault'])
y_test = test_data['isDefault']

# 初始化并训练模型
model = NaiveBayesClassifier()
model.fit(X_train, y_train)

# 进行预测
predictions = model.predict(X_test)

# 5. 计算分类准确度
accuracy = np.mean(predictions == y_test)
print("分类准确度：", accuracy)
