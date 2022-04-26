import pandas as pd
import numpy as np

dataset = pd.read_csv('datasets/50_Startups.csv')
X = dataset.iloc[:, : -1].values
Y = dataset.iloc[:, 4].values
print("X :", X)
# 将类别数字化
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
print("After LabelEncorder:\n", X)
ct = ColumnTransformer([("", OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()
print("After OnehotEncorder:\n", X)

# 躲避虚拟变量陷阱
X = X[:, 1:]

# 拆分数据集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 训练
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# 预测
y_pred = regressor.predict(X_test)
print('y_pred is : \n', y_pred)
print('y_true is : \n', Y_test)

# regression evaluation
from sklearn.metrics import r2_score

print('相似度:\n', r2_score(Y_test, y_pred))

# 绘图
import matplotlib.pyplot as plt

plt.scatter(np.arange(1, 11), Y_test, color='red')
plt.scatter(np.arange(1, 11), y_pred, color='blue')
plt.show()
