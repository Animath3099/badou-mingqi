import jieba
import synonyms
import random
from random import shuffle

random.seed(2019)

# 停用词列表，默认使用哈工大停用词表
f = open('../stopwords/hit_stopwords.txt', encoding='utf-8')
stop_words = list()
for stop_word in f.readlines():
    stop_words.append(stop_word[:-1])         # 一维列表，元素为停用词


# 考虑到与英文的不同，暂时搁置，文本清理
'''
import re
def get_only_chars(line):
    #1.清除所有的数字
'''


########################################################################
# 同义词替换，替换一个语句中的n个单词为其同义词
########################################################################
def synonym_replacement(words, n):
    new_words = words.copy()                       # 对于简单序列复制后，修改新值不会影响之前的值
    random_word_list = list(set([word for word in words if word not in stop_words]))     # 获取不属于停用词集的词
    random.shuffle(random_word_list)               # 搅乱顺序
    num_replaced = 0  
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)       # 调用函数，得到每个词的同义词
        if len(synonyms) >= 1:                     # 如果同义词比1个多，随机选择其中一个
            synonym = random.choice(synonyms)   
            new_words = [synonym if word == random_word else word for word in new_words]  # 用同义词替换之前的词，其他词保持不变
            num_replaced += 1                      # 替换数量+1
        if num_replaced >= n:                      # 仅替换n个词
            break

    sentence = ' '.join(new_words)                 # 用空格拼接同义词替换后的单词为句子字符串
    new_words = sentence.split(' ')                # 用空格分割句子，返回词列表

    return new_words


def get_synonyms(word):
    return synonyms.nearby(word)[0]      # 返回一维列表，元素为word的所有同义词


########################################################################
# 随机插入，随机在语句中插入n个词
########################################################################
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)           # 调用函数，执行n次插入操作
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0    
    while len(synonyms) < 1:                                            # 同义词列表小于1，则重复执行
        random_word = new_words[random.randint(0, len(new_words)-1)]    # 随机挑选一个单词
        synonyms = get_synonyms(random_word)                            # 调用函数，得到同义词列表
        counter += 1
        if counter >= 10:                                               # 如果重复挑选10个单词仍无法得到某个词的同义词，则跳过此次插入
            return
    random_synonym = random.choice(synonyms)                            # 随机选择同义词表中的一个词插入随机位置
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# 随机交换。随机交换句子中的两个单词n次
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)           # 调用函数，执行n次交换操作
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)        # 得到闭区间[0,..句子长度-1]的随机数
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)    # 得到闭区间[0,..句子长度-1]的随机数
        counter += 1
        if counter > 3:                                       # 如果随机三次仍然无法得到两个不同的索引，则不执行当前词的交换操作
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]   # 交换两个位置的词
    return new_words


########################################################################
# 随机删除，以概率p删除语句中的词
########################################################################
def random_deletion(words, p):
    if len(words) == 1:                  # 如果只有一个单词，则不执行删除操作
        return words

    new_words = []
    for word in words:                   # 以概率p随机删除单词
        r = random.uniform(0, 1)         # 均匀分布，闭区间[0,1]
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:                         # 如果最后删除了所有单词，只需返回words中的一个随机单词
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words


########################################################################
# EDA函数
def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    seg_list = jieba.cut(sentence)                  # 调用jieba分词
    seg_list = " ".join(seg_list)                   # 用空格拼接得到字符串
    words = list(seg_list.split())                  # 用空格分割得到一维列表
    num_words = len(words)                          # 词数量

    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1        # 每种增强方法对应生成增强句子数量，默认3
    n_sr = max(1, int(alpha_sr * num_words))        # 最少替换1个
    n_ri = max(1, int(alpha_ri * num_words))        # 最少替换1个
    n_rs = max(1, int(alpha_rs * num_words))        # 最少替换1个

    # print(words, "\n")
    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))
    # 随机插入ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))
    # 随机交换rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))
    # 随机删除rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(' '.join(a_words))
    # print(augmented_sentences)
    shuffle(augmented_sentences)
    # 修剪，得到想要的扩充句的数量
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)            # 每个增强句子保留概率
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    augmented_sentences.append(seg_list)   # 附加原句

    return augmented_sentences


if __name__ == '__main__':
    # 测试用例
    aug_sentences = eda(sentence="我们就像蒲公英，我也祈祷着能和你飞去同一片土地")
    print(aug_sentences)
