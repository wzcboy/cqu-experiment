import pandas as pd
import numpy as np

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