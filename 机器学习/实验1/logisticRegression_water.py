import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


# 载入数据集
excelPath = '/Users/mac/Downloads/the_project_xiguw.xlsx'
data = pd.read_excel(excelPath, 'Sheet1')
x1 = np.array(data[u'密度'])
x2 = np.array(data[u'含糖量'])
X = np.c_[x1, x2]
Y = np.array(data[u'好瓜'])
# print(X)
# print(Y)

# # 逻辑回归模型
# lr = LogisticRegression()
# lr.fit(X, Y)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# lr = LogisticRegression(C = 1e5) # C: Inverse of regularization strength
lr = LogisticRegression(C=1e5)
lr.fit(x_train, y_train)

print("Logistic Regression模型训练集的准确率：%.3f" %lr.score(x_train, y_train))
print("Logistic Regression模型测试集的准确率：%.3f" %lr.score(x_test, y_test))
# meshgrid函数生成两个网格矩阵
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# print(xx.shape)
# print(yy)
# print(xx.ravel().shape)

# pcolormesh函数将xx,yy两个网格矩阵和对应的预测结果Z绘制在图片上
# pcolormesh函数为分类常用画图函数
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])   # np.c_ 按照行连接两个矩阵
Z = Z.reshape(xx.shape)      # Z 只有0 ，1， 2三类数据

plt.figure(1, figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# 绘制散点图
plt.scatter(X[:8, 0], X[:8, 1], c='red', marker='o', label='good')
plt.scatter(X[8:, 0], X[8:, 1], c='blue', marker='x', label='bad')

# 坐标轴
plt.xlabel('density')
plt.ylabel('sugar')
plt.title('Watermelon Classification')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=2)
plt.show()