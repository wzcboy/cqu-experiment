import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math

# macos中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

data = pd.read_csv('概化后的数据.csv', encoding='utf-8')
print("原始数据集大小为：", data.shape)


def drawAreaDistribution():
    print("绘制公司地区分布条形图")
    areas = data['公司地区'].values
    area_dict = {}
    for a in areas:
        if a in area_dict.keys():
            area_dict[a] += 1
        else:
            area_dict[a] = 0
    area_x = []
    area_y = []
    for key, val in area_dict.items():
        area_x.append(key)
        area_y.append(val)
    plt.bar(area_x, area_y)
    plt.title('公司地区分布图')
    plt.xticks(rotation=90, fontsize=10)
    plt.show()


def drawAgeDistribution():
    print("绘制公司年龄分布条形图")
    age = data['公司年龄'].values
    n = len(age)
    # 组数 m = 1.87*(n-1)^0.4
    m = int(1.87 * math.pow(n-1, 0.4))
    print("组数为：", m)

    age = sorted(age)
    # 组间隔
    interval = (age[len(age) - 1] - age[0]) / m
    # 横坐标数值
    x_index = [age[0] + i * interval for i in range(m + 1)]
    # 绘制
    plt.hist(data['公司年龄'].values, x_index)
    plt.xlabel('公司年龄')
    plt.ylabel('频数')
    plt.title("公司年龄分布图")
    plt.show()


def drawWork():
    print("绘制工作类别分布扇形图")
    work_type = data['工作类别'].values
    work_type_dict = {}
    for w in work_type:
        if w in work_type_dict.keys():
            work_type_dict[w] += 1
        else:
            work_type_dict[w] = 0
    work_type_label = []
    work_type_x = []
    for key, val in work_type_dict.items():
        work_type_label.append(key)
        work_type_x.append(val)
    explode = (0, 0.1, 0, 0)
    plt.pie(work_type_x, explode=explode, labels=work_type_label, autopct='%1.1f%%', shadow=True)
    plt.title("工作类别分布扇形图")
    plt.show()


def drawCompanyType():
    print("绘制公司领域分布扇形图")
    work_type = data['公司领域'].values
    work_type_dict = {}
    for w in work_type:
        if w in work_type_dict.keys():
            work_type_dict[w] += 1
        else:
            work_type_dict[w] = 0
    work_type_label = []
    work_type_x = []
    for key, val in work_type_dict.items():
        work_type_label.append(key)
        work_type_x.append(val)
    explode = (0, 0.1, 0, 0, 0, 0)
    plt.pie(work_type_x, explode=explode, labels=work_type_label, autopct='%1.1f%%', shadow=True)
    plt.title("公司领域分布扇形图")
    plt.show()


def drawJobLevel():
    print("绘制职务等级分布扇形图")
    job_level = data['职务等级'].values
    job_level_dict = {}
    for w in job_level:
        if w in job_level_dict.keys():
            job_level_dict[w] += 1
        else:
            job_level_dict[w] = 0
    job_level_label = []
    job_level_x = []
    for key, val in job_level_dict.items():
        job_level_label.append(key)
        job_level_x.append(val)
    explode = (0, 0.1, 0, 0, 0)
    plt.pie(job_level_x, labels=job_level_label, autopct='%1.1f%%', shadow=True)
    plt.title("职务等级分布扇形图")
    plt.show()


def drawScatter():
    print("绘制散点图")
    plt.scatter(data["是否认证"].values, data["公司是否认证"].values, s=20, c="#ff1212", marker='o')
    plt.title("是否认证与公司是否认证的散点图")
    plt.xlabel("是否认证")
    plt.ylabel("公司是否认证")
    plt.show()

    plt.scatter(data["公司年龄"].values, data["注册资金"].values, s=20, c="#ff1212", marker='o')
    plt.title("公司年龄与公司注册资金的散点图")
    plt.xlabel("公司年龄")
    plt.ylabel("注册资金")
    plt.xlim(0, 40)
    plt.ylim(100000, 22000000000)
    plt.show()


def drawQQ():
    print("绘制QQ图")
    # Apply the default theme
    sns.set_theme()

    data1 = sorted(data["是否认证"].values)
    data2 = sorted(data["公司是否认证"].values)

    plt.xlabel("")
    plt.ylabel("")
    sns.regplot(x=data1, y=data2, color='pink')

    plt.show()

    data1 = sorted(data["公司年龄"].values)
    data2 = sorted(data["注册资金"].values)

    plt.xlabel("")
    plt.ylabel("")
    sns.regplot(x=data1, y=data2, color='pink')
    plt.xlim(0, 40)
    plt.ylim(100000, 22000000000)
    plt.show()


def compareAnalysis1():
    """
    职务等级的对比分析
    :return: none
    """
    print("进行数据的对比分析")
    keys = ['A', 'B', 'C', 'D', 'E']
    # 用户是否认证
    certification = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    uncertification = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    for i in data.index:
        level = data.loc[i, "职务等级"]
        isCertificated = data.loc[i, "是否认证"]
        if isCertificated:
            certification[level] += 1
        else:
            uncertification[level] += 1
    y1 = []
    y2 = []
    for key in keys:
        y1.append(certification[key])
        y2.append((uncertification[key]))
    x = np.arange(len(keys))  # the label locations
    width = 0.35                # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, y1, width, label='认证用户')
    rects2 = ax.bar(x + width / 2, y2, width, label='非认证用户')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('频数')
    ax.set_title('认证用户和非认证用户职务等级分布情况的差异')
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.legend()
    fig.tight_layout()

    plt.show()

    # 扇形图
    plt.subplot(1, 2, 1)
    plt.pie(y1, labels=keys, autopct='%1.1f%%')
    plt.title("认证用户职务等级分布情况")

    plt.subplot(1, 2, 2)
    plt.pie(y2, labels=keys, autopct='%1.1f%%')
    plt.title("非认证用户职务等级分布情况")

    plt.show()


def compareAnalysis2():
    """
    公司年龄的对比分析
    :return:
    """
    # 公司是否认证
    certification = []
    uncertification = []
    for i in data.index:
        age = data.loc[i, "公司年龄"]
        isCertificated = data.loc[i, "公司是否认证"]
        if isCertificated:
            certification.append(age)
        else:
            uncertification.append(age)

    plt.subplot(1, 2, 1)
    plt.boxplot(certification, flierprops={'marker': 'x'}, labels=['认证公司'], showmeans=True)
    plt.ylim(0, 70)
    plt.title("认证公司的公司年龄盒图")

    plt.subplot(1, 2, 2)
    plt.boxplot(uncertification, flierprops={'marker': 'x'}, labels=['非认证公司'], showmeans=True)
    plt.ylim(0, 70)
    plt.title("非认证公司的公司年龄盒图")
    plt.show()

    # ----------绘制认证公司年龄的条形图-----------
    n = len(certification)
    # 组数 m = 1.87*(n-1)^0.4
    m = int(1.87 * math.pow(n - 1, 0.4))
    print("组数为：", m)

    certification = sorted(certification)
    # 组间隔
    interval = (certification[len(certification) - 1] - certification[0]) / m
    # 横坐标数值
    x_index = [certification[0] + i * interval for i in range(m + 1)]
    # 绘制
    plt.hist(certification, x_index)
    plt.xlabel('公司年龄')
    plt.ylabel('频数')
    plt.title("认证的公司年龄分布图")
    plt.show()

    # ----------绘制非认证公司年龄的条形图-----------
    n = len(uncertification)
    # 组数 m = 1.87*(n-1)^0.4
    m = int(1.87 * math.pow(n - 1, 0.4))
    print("组数为：", m)

    uncertification = sorted(uncertification)
    # 组间隔
    interval = (uncertification[len(uncertification) - 1] - uncertification[0]) / m
    # 横坐标数值
    x_index = [uncertification[0] + i * interval for i in range(m + 1)]
    # 绘制
    plt.hist(uncertification, x_index)
    plt.xlabel('公司年龄')
    plt.ylabel('频数')
    plt.title("非认证的公司年龄分布图")
    plt.show()


def compareAnalysis3():
    """
    公司注册资金的对比分析
    :return:
    """
    # 公司是否认证
    certification = []
    uncertification = []
    for i in data.index:
        age = data.loc[i, "注册资金"]
        isCertificated = data.loc[i, "公司是否认证"]
        if isCertificated:
            certification.append(age)
        else:
            uncertification.append(age)

    plt.subplot(1, 2, 1)
    plt.boxplot(certification, flierprops={'marker': 'x'}, labels=['认证公司'], showmeans=True)
    plt.title("认证公司的注册资金盒图")

    plt.subplot(1, 2, 2)
    plt.boxplot(uncertification, flierprops={'marker': 'x'}, labels=['非认证公司'], showmeans=True)
    plt.title("非认证公司的注册资金盒图")
    plt.show()

    # ----------绘制认证公司注册资金的条形图-----------
    n = len(certification)
    # 组数 m = 1.87*(n-1)^0.4
    m = int(1.87 * math.pow(n - 1, 0.4))
    print("组数为：", m)

    certification = sorted(certification)
    # 组间隔
    interval = (certification[len(certification) - 1] - certification[0]) / m
    # 横坐标数值
    x_index = [certification[0] + i * interval for i in range(m + 1)]
    # 绘制
    plt.hist(certification, x_index)
    plt.xlabel('注册资金')
    plt.ylabel('频数')
    plt.title("认证的公司注册资金分布图")
    plt.xlim(0, 42000000000)
    plt.show()

    # ----------绘制非认证公司注册资金的条形图-----------
    n = len(uncertification)
    # 组数 m = 1.87*(n-1)^0.4
    m = int(1.87 * math.pow(n - 1, 0.4))
    print("组数为：", m)

    uncertification = sorted(uncertification)
    # 组间隔
    interval = (uncertification[len(uncertification) - 1] - uncertification[0]) / m
    # 横坐标数值
    x_index = [uncertification[0] + i * interval for i in range(m + 1)]
    # 绘制
    plt.hist(uncertification, x_index)
    plt.xlabel('注册资金')
    plt.ylabel('频数')
    plt.title("非认证的公司注册资金分布图")
    plt.xlim(0, 42000000000)
    plt.show()

def compareAnalysis4():
    """
    注册资金等级的对比分析
    :return: none
    """
    keys = ['1000万以下', '1000万以上5000万以下', '5000万以上1亿以下', '1亿以上']
    # 公司是否认证
    certification = {'1000万以下': 0, '1000万以上5000万以下': 0, '5000万以上1亿以下': 0, '1亿以上': 0}
    uncertification = {'1000万以下': 0, '1000万以上5000万以下': 0, '5000万以上1亿以下': 0, '1亿以上': 0}
    for i in data.index:
        level = data.loc[i, "注册资金等级"]
        isCertificated = data.loc[i, "公司是否认证"]
        if isCertificated:
            certification[level] += 1
        else:
            uncertification[level] += 1
    y1 = []
    y2 = []
    for key in keys:
        y1.append(certification[key])
        y2.append((uncertification[key]))
    x = np.arange(len(keys))  # the label locations
    width = 0.35                # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, y1, width, label='认证公司')
    rects2 = ax.bar(x + width / 2, y2, width, label='非认证公司')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('频数')
    ax.set_title('认证公司和非认证公司注册资金等级分布情况的差异')
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.legend()
    fig.tight_layout()

    plt.show()

    # 扇形图
    plt.subplot(1, 2, 1)
    plt.pie(y1, labels=keys, autopct='%1.1f%%')
    plt.title("认证公司注册资金等级分布情况")

    plt.subplot(1, 2, 2)
    plt.pie(y2, labels=keys, autopct='%1.1f%%')
    plt.title("非认证公司注册资金等级分布情况")

    plt.show()


if __name__ == '__main__':
    print("------------开始进行数据可视化-------------")
    # drawAreaDistribution()
    # drawAgeDistribution()
    drawCompanyType()
    # drawWorkType()
    # drawJobLevel()
    # drawScatter()
    # drawQQ()
    # compareAnalysis1()
    # compareAnalysis2()
    # compareAnalysis3()
    # compareAnalysis4()