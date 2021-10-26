import re

TRAIN_FILE = '../data/toutiao_cat_data.txt'

SENTENCE_FILE = './result/setence.txt'



def process_news_data(sentence_list):
    with open(TRAIN_FILE) as f:
        count = 0
        for line in f.readlines():
            # remind
            count += 1
            if count % 10000 == 0:
                print(count)

            content_list = line.strip().split('_!_')

            # 新闻主要内容
            content3= content_list[3]
            # 将新闻主要内容按照标点符号进行切分
            sentences = re.split(r"(，|,|。|；|？|！|：)", content3.strip())
            # 去除句子列表中长度小于等于1的，认为这不是一个句子
            sentences = list(filter(lambda x: len(x) > 1 and is_chinese(x), sentences))
            # example ['上联', '山水醉人何须酒', '如何对下联']
            # print(sentences)
            # 将句子添加到结果
            sentence_list += sentences


            # 新闻关键字
            content4 = content_list[4]
            # 去除空字符串
            if len(content4) < 1:
                continue
            # 将新闻关键词按照标点符号进行切分
            keywords = re.split(r"(，|,|。|；|？|！|：)", content4.strip())
            # 去除句子列表中长度小于等于1的，认为这不是一个句子
            keywords = list(filter(lambda x: len(x) > 1 and is_chinese(x), keywords))
            # example ['保利集团', '马未都', '中国科学技术馆']
            # print(keywords)
            # 将句子添加到结果
            sentence_list += keywords


def is_chinese(v):
    if len(v) == 0:
        return False
    return all(u'\u4e00' <= c <= u'\u9fff' or c == u'〇' for c in v)


def save_sentence(sentences, file_path):
    print("要写入文件的列表长度为：", len(sentence_list))
    with open(file_path, 'w') as f:
        for sentence in sentences:
            f.write(sentence)
            f.write('\n')
        print("写入文件" + file_path + "成功")


if __name__ == '__main__':
    sentence_list = []
    process_news_data(sentence_list)
    save_sentence(sentence_list, SENTENCE_FILE)
