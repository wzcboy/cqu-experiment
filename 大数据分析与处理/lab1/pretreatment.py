import pandas as pd
import numpy as np
import re

data = pd.read_csv('数据集_找到网人力资源数据2.csv', encoding='gb18030')
print("原始数据集大小为：", data.shape)


def digitalizeAge():
    """
    将公司年龄处理为数值类型，并且缺失值填充为平均值
    相应更改公司成立时间
    :return: none
    """
    ageSum = 0
    cnt = 0
    for i in data.index:
        if pd.isna(data.loc[i, "公司年龄"]):
            continue
        cnt += 1
        if data.loc[i, "公司年龄"] == "不足一年":
            data.loc[i, "公司年龄"] = 0
        else:
            num = int(data.loc[i, "公司年龄"][:-1])
            data.loc[i, "公司年龄"] = num
            ageSum += num

    # 计算中位数并将缺失值填充
    average = int(ageSum/cnt)
    print("平均公司年龄为:", average)
    for i in data.index:
        if pd.isna(data.loc[i, "公司年龄"]):
            data.loc[i, "公司年龄"] = average
            data.loc[i, "公司成立时间"] = str(2018-average) + "/1/1"


def digitalizeFund():
    """
    将注册资金处理为数值类型，并且缺失值填充为中位数
    :return: none
    """
    fundList = []
    for i in data.index:
        if pd.isna(data.loc[i, "注册资金"]):
            continue
        num = float(data.loc[i, "注册资金"][:-1])
        if data.loc[i, "注册资金"][-1] == '万':
            num = num * 10000
        elif data.loc[i, "注册资金"][-1] == '亿':
            num = num * 100000000
        # 更新
        data.loc[i, "注册资金"] = int(num)
        fundList.append(int(num))

    # 计算中位数并将缺失值填充
    median = int(np.median(fundList))
    print("注册资金中位数:", median)
    for i in data.index:
        if pd.isna(data.loc[i, "注册资金"]):
            data.loc[i, "注册资金"] = median


def validateTel():
    """
    统计不符合格式的电话号码，并将其置为nan
    :return: none
    """
    # 统计不符合格式的电话号码
    cnt = 0
    for i in data.index:
        if pd.isna(data.loc[i, "tel"]):
            continue
        temp = data.loc[i, "tel"] / 10000000000
        if temp <= 1 or temp >= 2:
            data.loc[i, "tel"] = np.nan
            cnt += 1
    print("电话不符合格式的有{}条".format(cnt))


def removeNosense():
    """
    删除条目中姓名为空或者带有数字，或者职务为空，或者公司名称为空的数据
    :return: none
    """
    cnt = 0
    pattern = re.compile('[0-9]+')
    for i in data.index:
        if pd.isna(data.loc[i, "部门-职务"]) or pd.isna(data.loc[i, "公司名称"]) or pd.isna(data.loc[i, "姓名"]):
            data.drop([i], inplace=True)
            cnt += 1
            continue
        match1 = pattern.findall(data.loc[i, "姓名"])
        match2 = pattern.findall(data.loc[i, "部门-职务"])
        match3 = pattern.findall(data.loc[i, "公司名称"])
        if match1 or match2 or match3:
            data.drop([i], inplace=True)
            cnt += 1

    data.reset_index(drop=True, inplace=True)
    print("删除的条目有{}".format(cnt))


def standardizeCompanyTime():
    """
    将公司时间标准化
    从 2017/1/1 -> 2017-01-01
    :return: none
    """
    print("将公司成立时间标准化")
    for i in data.index:
        if pd.isna(data.loc[i, "公司成立时间"]):
            continue
        timeList = str(data.loc[i, "公司成立时间"]).split('/')
        if len(timeList) != 3:
            print("wrong", i, timeList)
            continue
        year = str(timeList[0])
        # 前导0填充
        month = str(timeList[1]).zfill(2)
        day = str(timeList[2]).zfill(2)
        data.loc[i, "公司成立时间"] = year + '-' + month + '-' + day


def splitEducation():
    """
    将教育经历拆分为：学校+专业+开始结束时间
    教育经历有四种情况：（1）只有学校；（2）有学校和专业；（3）有学校和开始结束时间；（4）学习，专业，开始结束时间都有
    :return: none
    """
    school = []
    major = []
    time = []
    pattern = re.compile('^(.*?) \d((.*?|$)((\d{4}-\d{2}-\d{2}) (\d{4}-\d{2}-\d{2}|\S{2})|$)|$)$')
    for i in data.index:
        if pd.isna(data.loc[i, "教育经历"]):
            school.append("")
            major.append("")
            time.append("")
            continue
        # 有可能有多段教育经历
        education_list = str(data.loc[i, "教育经历"]).split('|')
        school_item = ""
        major_item = ""
        time_item = ""
        for j in range(len(education_list)):
            match = re.match(pattern, education_list[j].strip())
            if match.group(5) == None and match.group(6) == None:
                if match.group(3) == None or match.group(3).strip() == '':
                    school_item += str(match.group(1))
                else:
                    school_item += str(match.group(1))
                    major_item += str(match.group(3))
            else:
                if match.group(3) == None or match.group(3).strip() == '':
                    school_item += str(match.group(1))
                    time_item += str(match.group(5) + '~' + match.group(6))
                else:
                    school_item += str(match.group(1))
                    major_item += str(match.group(3))
                    time_item += str(match.group(5)+'~'+match.group(6))
            # 不同的经历之间以 | 分隔
            if j != (len(education_list) - 1):
                school_item += '|'
                if major_item != '' and major_item[-1] != '|':
                    major_item += '|'
                if time_item != '' and time_item[-1] != '|':
                    time_item += '|'
        if school_item != '' and school_item.strip()[-1] == '|':
            school_item = school_item.strip()[:-1]
        if major_item != '' and major_item.strip()[-1] == '|':
            major_item = major_item.strip()[:-1]
        if time_item != '' and time_item.strip()[-1] == '|':
            time_item = time_item.strip()[:-1]

        school.append(school_item)
        major.append(major_item)
        time.append(time_item)
    # 插入新的三个column，并删除旧的
    data['学校'] = school
    data['专业'] = major
    data['入学时间-毕业时间'] = time
    data.drop(['教育经历'], axis=1, inplace=True)


def splitDepartAndJob():
    """
    将 部门-职务 拆分开来
    :return: none
    """
    depart = []
    job = []
    for i in data.index:
        departAndJob = str(data.loc[i, "部门-职务"]).strip().split()
        if len(departAndJob) == 0:
            depart.append("")
            job.append("")
            continue
        if len(departAndJob) > 2:
            # print(i, departAndJob)
            depart.append(departAndJob[0])
            job.append(departAndJob[len(departAndJob)-1])
        elif len(departAndJob) == 2:
            depart.append(departAndJob[0])
            job.append(departAndJob[1])
        else:
            depart.append("无")
            job.append(departAndJob[0])
    data.insert(2, '部门', depart)
    data.insert(3, '职务', job)
    data.drop(['部门-职务'], axis=1, inplace=True)


if __name__ == '__main__':
    removeNosense()

    validateTel()

    digitalizeFund()
    digitalizeAge()
    standardizeCompanyTime()

    splitEducation()
    splitDepartAndJob()

    # 重新写入新的csv文件
    # data.to_csv('人力资源预处理后的数据.csv')