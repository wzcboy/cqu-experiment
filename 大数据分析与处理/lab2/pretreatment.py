import pandas as pd
import numpy as np
from bisect import bisect_left
from sklearn.model_selection import train_test_split

# 读取文件数据
f = open('ratings.dat', encoding='utf-8')
# 存放数据的二维数组
dataList = []

for line in f:
    s = line.strip().split('::')
    data = []
    # 字符串转化为整数
    for i in range(len(s)-1):
        data.append(int(s[i]))
    dataList.append(data)

# 关闭文件
print('文件读取完毕')
f.close()

# 形成DataFrame
df = pd.DataFrame(dataList, columns=['UserID', 'MovieID', 'Rating'])

# 统计数据
userID_min = min(df['UserID'].values)     # 1
userID_max = max(df['UserID'].values)     # 6040
cnt1 = len(df['UserID'].unique())         # 6040

movieID_min = min(df['MovieID'].values)   # 1
movieID_max = max(df['MovieID'].values)   # 3952
cnt2 = len(df['MovieID'].unique())        # 3706

# 用户字典{userID: movieNum}
dataDict = df['UserID'].value_counts()
# 排好序的电影ID数组和用户ID数组
movieID_sorted = sorted(df['MovieID'].unique())
userID_sorted = sorted(df['UserID'].unique())

start = 0

trainMatrix = np.zeros((6040, 3706))
valMatrix = np.zeros((6040, 3706))
testMatrix = np.zeros((6040, 3706))

# 对每一个用户遍历
for i in range(1, len(dataDict)+1):
    cnt = dataDict[i]
    end = start + cnt
    # 划分训练集，验证集，测试集
    X_train, X_remain = train_test_split(dataList[start:end], test_size=0.2, random_state=42)
    X_test, X_validate = train_test_split(X_remain, test_size=0.5, random_state=42)

    for t in X_train:
        # 二分查找
        index = bisect_left(movieID_sorted, t[1])
        trainMatrix[i-1][index] = t[2]

    for t in X_validate:
        # 二分查找
        index = bisect_left(movieID_sorted, t[1])
        valMatrix[i-1][index] = t[2]

    for t in X_test:
        # 二分查找
        index = bisect_left(movieID_sorted, t[1])
        testMatrix[i - 1][index] = t[2]
    # update start
    start = end

# 存储为文件形式
np.save("训练集矩阵.npy", trainMatrix)
np.save("验证集矩阵.npy", valMatrix)
np.save("测试集矩阵.npy", testMatrix)

np.save("用户列表.npy", userID_sorted)
np.save("电影列表.npy", movieID_sorted)