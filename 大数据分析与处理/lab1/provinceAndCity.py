"""
通过国家统计局数据
获取中国所有城市列表
"""
import sys
import os
import re
from urllib import request
from bs4 import BeautifulSoup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
url = 'http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2020/'
header = {
    'Cookie': 'SF_cookie_1=15502425',
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Mobile Safari/537.36'}


class GetHttp:
    def __init__(self, url, headers=None, charset='utf8'):
        if headers is None:
            headers = {}
        self._response = ''
        try:
            print(url)
            self._response = request.urlopen(request.Request(url=url, headers=headers))
        except Exception as e:
            print(e)
        self._c = charset

    @property
    def text(self):
        try:
            return self._response.read().decode(self._c)
        except Exception as e:
            print(e)
            return ''


def provincetr(u, he, lists):
    # 获取全国省份和直辖市
    t = GetHttp(u, he, 'gbk').text
    if t:
        soup = BeautifulSoup(t, 'html.parser')
        for i in soup.find_all(attrs={'class': 'provincetr'}):
            for a in i.find_all('a'):
                id = re.sub("\D", "", a.get('href'))
                lists[id] = a.text
                # time.sleep(1 / 10)
    return lists


def citytr(u, he, lists):
    # 获取省下级市
    l = lists.copy()
    for i in l:
        t = GetHttp(u+i+'.html', he, 'gbk').text
        if not t:
            continue
        soup = BeautifulSoup(t, 'html.parser')
        for v in soup.find_all(attrs={'class': 'citytr'}):
            id = str(v.find_all('td')[0].text)
            if id[0:4] not in lists.keys():
                lists[id[0:4]] = str(v.find_all('td')[1].text)

    return lists


def countytr(u, he, lists):
    # 获取市下级县
    l = lists.copy()
    a = {}
    for i in l:
        if len(i) != 4:
            continue
        t = GetHttp(u+i[0:2]+'/'+i+'.html', he, 'gbk').text
        if not t:
            continue
        soup = BeautifulSoup(t, 'html.parser')
        for v in soup.find_all(attrs={'class': 'countytr'}):
            id = str(v.find_all('td')[0].text)
            if id[0:6] not in lists.keys():
                lists[id[0:6]] = {'id': id[0:6], 'name': str(v.find_all('td')[1].text)}
    return lists


def towntr(u, he, lists):
    # 县下级镇
    l = lists.copy()
    for i in l:
        t = GetHttp(u+i[0:2]+'/'+i[2:4]+'/'+i+'.html', he, 'gbk').text
        if not t:
            continue
        soup = BeautifulSoup(t, 'html.parser')
        for v in soup.find_all(attrs={'class': 'towntr'}):
            id = str(v.find_all('td')[0].text)
            if id[0:9] not in lists.keys():
                lists[id[0:9]] = {'id': id[0:9], 'name': str(v.find_all('td')[1].text), 'pid': '0',
                                  'pid1': l[i]['pid1'], 'pid2': l[i]['pid2'], 'pid3': i, 'pid4': '0', 'code': id}
    return lists


def villagetr(u, he, lists):
    # 镇下级村
    l = lists.copy()
    for i in l:
        t = GetHttp(u+i[0:2]+'/'+i[2:4]+'/'+i[4:6]+'/'+i+'.html', he, 'gbk').text
        if not t:
            continue
        soup = BeautifulSoup(t, 'html.parser')
        for v in soup.find_all(attrs={'class': 'villagetr'}):
            id = str(v.find_all('td')[0].text)
            if id[0:12] not in lists.keys():
                lists[id[0:12]] = {'id': id[0:12], 'name': str(v.find_all('td')[1].text), 'pid': '0',
                                   'pid1': l[i]['pid1'], 'pid2': l[i]['pid2'], 'pid3': l[i]['pid2'], 'pid4': i,
                                   'code': id}
    return lists


if __name__ == '__main__':
    p = provincetr(u=url, he=header, lists={})
    print('省')
    c = citytr(u=url, he=header, lists=p)
    print('市')
    o = countytr(u=url, he=header, lists=c)
    print('县')
    # t = towntr(u=url, he=header, lists=o)
    # print('镇')
    # v = villagetr(u=url, he=header, lists=t)
    # print('村')
