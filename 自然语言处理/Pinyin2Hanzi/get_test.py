TEST_FILE = '../data/test.txt'


def get_test_data():
    # 划分拼音和汉语句子
    pinyin_list = []
    hanzi_list = []
    with open(TEST_FILE) as f:
        content = f.readlines()
        for i in range(len(content)):
            if i % 2 == 0:
                pinyin_list.append(content[i].strip())
            else:
                hanzi_list.append(content[i].strip())
    return pinyin_list, hanzi_list


pinyin_list, hanzi_list = get_test_data()
print("---------标准结果---------")
pinyin_list, hanzi_list = get_test_data()
for state in hanzi_list:
    state = state.strip()
    print(state)

