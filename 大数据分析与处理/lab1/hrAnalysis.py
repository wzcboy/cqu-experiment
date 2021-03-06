import pandas as pd
import numpy as np
import scipy
import re

data = pd.read_csv('/Users/mac/Desktop/大数据分析与处理作业/人力资源预处理后的数据.csv', encoding='utf-8')
print("预处理后数据集大小为：", data.shape)

job_grades = {'A': ['董事', '主席', '创始人', '股东'], 'B': ['总经理', '总裁', '副总经理'],
              'C': ['总监', '副总监', '经理', '副经理', '主任', '主管', '代表', '副主任', '主任', '科长', '副科长', '课长'],
              'D': ['工程师', '实习生', '员', '经纪人', '顾问', '秘书', '助理', '规划师']}

work_types = {'市场类': ['销售', '市场', '客户', '招商'], '技术类': ['业务', '技术', '项目', '研发'],
             '营销类': ['营销', '宣传', '渠道'], '其他类': ['财务', '运营', '行政', '人力']}

company_types = {'科技类': ['科技', '软件', '信息技术'], '文化传媒广告类': ['文化', '传媒', '广告'],
             '咨询类': ['咨询'], '管理类': ['管理'], '贸易类': ['贸易', '商贸', '科贸', '工贸'],
                 '其他类': ['机械', '设备', '建筑', '人力']}

areas = {'11': '北京', '12': '天津', '13': '河北', '14': '山西', '15': '内蒙古自治区', '21': '辽宁', '22': '吉林',
         '23': '黑龙江', '31': '上海', '32': '江苏', '33': '浙江', '34': '安徽', '35': '福建', '36': '江西',
         '37': '山东', '41': '河南', '42': '湖北', '43': '湖南', '44': '广东', '45': '广西壮族自治区', '46': '海南',
         '50': '重庆', '51': '四川', '52': '贵州', '53': '云南', '54': '西藏自治区', '61': '陕西', '62': '甘肃',
         '63': '青海', '64': '宁夏回族自治区', '65': '新疆维吾尔自治区', '1101': '辖区', '1201': '辖区', '1301': '石家庄',
         '1302': '唐山', '1303': '秦皇岛', '1304': '邯郸', '1305': '邢台', '1306': '保定', '1307': '张家口',
         '1308': '承德', '1309': '沧州', '1310': '廊坊', '1311': '衡水', '1401': '太原', '1402': '大同',
         '1403': '阳泉', '1404': '长治', '1405': '晋城', '1406': '朔州', '1407': '晋中', '1408': '运城',
         '1409': '忻州', '1410': '临汾', '1411': '吕梁', '1501': '呼和浩特', '1502': '包头', '1503': '乌海',
         '1504': '赤峰', '1505': '通辽', '1506': '鄂尔多斯', '1507': '呼伦贝尔', '1508': '巴彦淖尔', '1509': '乌兰察布',
         '1522': '兴安盟', '1525': '锡林郭勒盟', '1529': '阿拉善盟', '2101': '沈阳', '2102': '大连', '2103': '鞍山',
         '2104': '抚顺', '2105': '本溪', '2106': '丹东', '2107': '锦州', '2108': '营口', '2109': '阜新',
         '2110': '辽阳', '2111': '盘锦', '2112': '铁岭', '2113': '朝阳', '2114': '葫芦岛', '2201': '长春',
         '2202': '吉林', '2203': '四平', '2204': '辽源', '2205': '通化', '2206': '白山', '2207': '松原',
         '2208': '白城', '2224': '延边朝鲜族自治州', '2301': '哈尔滨', '2302': '齐齐哈尔', '2303': '鸡西',
         '2304': '鹤岗', '2305': '双鸭山', '2306': '大庆', '2307': '伊春', '2308': '佳木斯', '2309': '七台河',
         '2310': '牡丹江', '2311': '黑河', '2312': '绥化', '2327': '大兴安岭地区', '3101': '辖区', '3201': '南京',
         '3202': '无锡', '3203': '徐州', '3204': '常州', '3205': '苏州', '3206': '南通', '3207': '连云港',
         '3208': '淮安', '3209': '盐城', '3210': '扬州', '3211': '镇江', '3212': '泰州', '3213': '宿迁',
         '3301': '杭州', '3302': '宁波', '3303': '温州', '3304': '嘉兴', '3305': '湖州', '3306': '绍兴',
         '3307': '金华', '3308': '衢州', '3309': '舟山', '3310': '台州', '3311': '丽水', '3401': '合肥',
         '3402': '芜湖', '3403': '蚌埠', '3404': '淮南', '3405': '马鞍山', '3406': '淮北', '3407': '铜陵',
         '3408': '安庆', '3410': '黄山', '3411': '滁州', '3412': '阜阳', '3413': '宿州', '3415': '六安',
         '3416': '亳州', '3417': '池州', '3418': '宣城', '3501': '福州', '3502': '厦门', '3503': '莆田',
         '3504': '三明', '3505': '泉州', '3506': '漳州', '3507': '南平', '3508': '龙岩', '3509': '宁德',
         '3601': '南昌', '3602': '景德镇', '3603': '萍乡', '3604': '九江', '3605': '新余', '3606': '鹰潭',
         '3607': '赣州', '3608': '吉安', '3609': '宜春', '3610': '抚州', '3611': '上饶', '3701': '济南',
         '3702': '青岛', '3703': '淄博', '3704': '枣庄', '3705': '东营', '3706': '烟台', '3707': '潍坊',
         '3708': '济宁', '3709': '泰安', '3710': '威海', '3711': '日照', '3713': '临沂', '3714': '德州',
         '3715': '聊城', '3716': '滨州', '3717': '菏泽', '4101': '郑州', '4102': '开封', '4103': '洛阳',
         '4104': '平顶山', '4105': '安阳', '4106': '鹤壁', '4107': '新乡', '4108': '焦作', '4109': '濮阳',
         '4110': '许昌', '4111': '漯河', '4112': '三门峡', '4113': '南阳', '4114': '商丘', '4115': '信阳',
         '4116': '周口', '4117': '驻马店', '4190': '直辖县级行政区划', '4201': '武汉', '4202': '黄石',
         '4203': '十堰', '4205': '宜昌', '4206': '襄阳', '4207': '鄂州', '4208': '荆门', '4209': '孝感',
         '4210': '荆州', '4211': '黄冈', '4212': '咸宁', '4213': '随州', '4228': '恩施土家族苗族自治州',
         '4290': '直辖县级行政区划', '4301': '长沙', '4302': '株洲', '4303': '湘潭', '4304': '衡阳',
         '4305': '邵阳', '4306': '岳阳', '4307': '常德', '4308': '张家界', '4309': '益阳', '4310': '郴州',
         '4311': '永州', '4312': '怀化', '4313': '娄底', '4331': '湘西土家族苗族自治州', '4401': '广州',
         '4402': '韶关', '4403': '深圳', '4404': '珠海', '4405': '汕头', '4406': '佛山', '4407': '江门',
         '4408': '湛江', '4409': '茂名', '4412': '肇庆', '4413': '惠州', '4414': '梅州', '4415': '汕尾',
         '4416': '河源', '4417': '阳江', '4418': '清远', '4419': '东莞', '4420': '中山', '4451': '潮州',
         '4452': '揭阳', '4453': '云浮', '4501': '南宁', '4502': '柳州', '4503': '桂林', '4504': '梧州',
         '4505': '北海', '4506': '防城港', '4507': '钦州', '4508': '贵港', '4509': '玉林', '4510': '百色',
         '4511': '贺州', '4512': '河池', '4513': '来宾', '4514': '崇左', '4601': '海口', '4602': '三亚',
         '4603': '三沙', '4604': '儋州', '4690': '直辖县级行政区划', '5001': '辖区', '5002': '县', '5101': '成都',
         '5103': '自贡', '5104': '攀枝花', '5105': '泸州', '5106': '德阳', '5107': '绵阳', '5108': '广元',
         '5109': '遂宁', '5110': '内江', '5111': '乐山', '5113': '南充', '5114': '眉山', '5115': '宜宾',
         '5116': '广安', '5117': '达州', '5118': '雅安', '5119': '巴中', '5120': '资阳', '5132': '阿坝藏族羌族自治州',
         '5133': '甘孜藏族自治州', '5134': '凉山彝族自治州', '5201': '贵阳', '5202': '六盘水', '5203': '遵义',
         '5204': '安顺', '5205': '毕节', '5206': '铜仁', '5223': '黔西南布依族苗族自治州', '5226': '黔东南苗族侗族自治州',
         '5227': '黔南布依族苗族自治州', '5301': '昆明', '5303': '曲靖', '5304': '玉溪', '5305': '保山',
         '5306': '昭通', '5307': '丽江', '5308': '普洱', '5309': '临沧', '5323': '楚雄彝族自治州',
         '5325': '红河哈尼族彝族自治州', '5326': '文山壮族苗族自治州', '5328': '西双版纳傣族自治州', '5329': '大理白族自治州',
         '5331': '德宏傣族景颇族自治州', '5333': '怒江傈僳族自治州', '5334': '迪庆藏族自治州', '5401': '拉萨',
         '5402': '日喀则', '5403': '昌都', '5404': '林芝', '5405': '山南', '5406': '那曲', '5425': '阿里地区',
         '6101': '西安', '6102': '铜川', '6103': '宝鸡', '6104': '咸阳', '6105': '渭南', '6106': '延安',
         '6107': '汉中', '6108': '榆林', '6109': '安康', '6110': '商洛', '6201': '兰州', '6202': '嘉峪关',
         '6203': '金昌', '6204': '白银', '6205': '天水', '6206': '武威', '6207': '张掖', '6208': '平凉',
         '6209': '酒泉', '6210': '庆阳', '6211': '定西', '6212': '陇南', '6229': '临夏回族自治州',
         '6230': '甘南藏族自治州', '6301': '西宁', '6302': '海东', '6322': '海北藏族自治州', '6323': '黄南藏族自治州',
         '6325': '海南藏族自治州', '6326': '果洛藏族自治州', '6327': '玉树藏族自治州', '6328': '海西蒙古族藏族自治州',
         '6401': '银川', '6402': '石嘴山', '6403': '吴忠', '6404': '固原', '6405': '中卫', '6501': '乌鲁木齐',
         '6502': '克拉玛依', '6504': '吐鲁番', '6505': '哈密', '6523': '昌吉回族自治州', '6527': '博尔塔拉蒙古自治州',
         '6528': '巴音郭楞蒙古自治州', '6529': '阿克苏地区', '6530': '克孜勒苏柯尔克孜自治州', '6531': '喀什地区',
         '6532': '和田地区', '6540': '伊犁哈萨克自治州', '6542': '塔城地区', '6543': '阿勒泰地区', '6590': '自治区直辖县级行政区划'}


def judge(s, levelList, default):
    """
    判断s是否包含levelList中的某个值
    :param s: 输入的字符串
    :param levelList: 字典
    :param default: 没有找到的默认值
    :return: 关键字
    """
    for key, val_list in levelList.items():
        for val in val_list:
            if val in s:
                return key
    return default


def judgeFund(num):
    if num >= 100000000:
        return "1亿以上"
    elif num >= 50000000:
        return  "5000万以上1亿以下"
    elif num >= 10000000:
        return "1000万以上5000万以下"
    else:
        return "1000万以下"


def judgeArea(s):
    """
    根据公司名称判断它所在的省和直辖市
    :param s:
    :return:
    """
    for key, val in areas.items():
        if val in s:
            return areas[key[:2]]
    return "未知"


def gradeJob():
    """
    对职务进行概化，分为A，B，C，D，E五个等级
    :return: none
    """
    job_level = []

    for i in data.index:
        job = data.loc[i, '职务']
        level = judge(job, job_grades, 'E')
        job_level.append(level)
    data.insert(4, '职务等级', job_level)
    print("职务等级概化完毕")


def gradeWork():
    """
    对部门进行概化，分为市场类（关键词“销售/市场/客户”等）、技术类（关键词“业务/技术/项目”等）、营销类（关键词“营销/宣传”等）、
    其他类（关键词“财务/运营/行政/人力”等）
    :return:none
    """
    work_level = []

    for i in data.index:
        depart = data.loc[i, '部门']
        job = data.loc[i, '职务']
        info = depart + ' ' + job
        level = judge(info, work_types, '其他类')
        work_level.append(level)
    data.insert(3, '工作类别', work_level)
    print("工作类别概化完毕")


def gradeCompany():
    """
    根据公司名称，对公司所属领域进行概化
    分为科技类（关键词“科技/软件/信息技术”等），文化、传媒广告类（关键词“文化/传媒/广告”等）、咨询类（关键词“咨询”等）、
    管理类（关键词“管理”等）、贸易类（关键词“贸易/商贸/科贸/工贸”等）与其他类（关键词“机械/设备/建筑”等）
    :return: none
    """
    work_level = []

    for i in data.index:
        company_name = data.loc[i, '公司名称']
        level = judge(company_name, company_types, '其他类')
        work_level.append(level)
    data.insert(8, '公司领域', work_level)
    print("公司领域概化完毕")


def gradeFund():
    """
    对注册资金进行概化处理，分为“1000万以下”、“1000万以上5000万以下”、“5000万以上1亿以下”、“1亿以上”
    :return: none
    """
    fund_level = []
    for i in data.index:
        fund = data.loc[i, '注册资金']
        level = judgeFund(int(fund))
        fund_level.append(level)
    data.insert(15, '注册资金等级', fund_level)
    print("注册资金概化完毕")


def divideArea():
    """
    根据公司名称来决定它所在的省和直辖市
    :return:
    """
    location = []
    for i in data.index:
        company_name = data.loc[i, '公司名称']
        loc = judgeArea(company_name)
        location.append(loc)
    data.insert(7, '公司地区', location)
    print("公司区域概化完毕")


if __name__ == '__main__':
    print("------------开始进行数据概化------------")
    gradeJob()
    gradeWork()
    divideArea()
    gradeCompany()
    gradeFund()

    # 重新写入新的csv文件
    data.to_csv('概化后的数据.csv', index=False)