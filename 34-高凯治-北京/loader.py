import json
import re
import os
import torch
import random
import jieba
import numpy as np
from config import Config
from torch.utils.data import DataLoader
"""
数据加载
"""


class Dataset:
    def __init__(self, data_path, config):
        self.config = config                                 # 配置信息
        self.path = data_path                                # 数据路径
        self.vocab = load_vocab(config["vocab_path"])        # 调用函数，加载词典
        self.config["vocab_size"] = len(self.vocab)          # 词典长度
        self.sentences = []                                  # 一维列表，每个元素为从数据文件中加载的一句话（字符串）
        self.schema = self.load_schema(config["schema_path"])  # 调用函数，加载标签
        self.config["class_num"] = len(self.schema)          # 标签数量
        self.max_length = config["max_length"]               # 最大长度
        self.data = []
        self.load()

    def load(self):
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n")                   # 打开数据集文件，用换行符分割，返回一维列表，每个元素代表一段话
            for segment in segments:                          # 循环每段话
                if segment.strip() == "":                     # 如果为空字符则跳到下一段落
                    continue
                labels = []                                   # 存放一段话中每个字符的标签
                for i in range(len(segment)-1):
                    if segment[i] == '，' or segment[i] == '。' or segment[i] == '？':
                        pass
                    elif segment[i+1] == '，':
                        labels.append(self.schema['，'])
                    elif segment[i+1] == '。':
                        labels.append(self.schema['。'])
                    elif segment[i+1] == '？':
                        labels.append(self.schema['？'])
                    else:
                        labels.append(self.schema[''])
                if segment[len(segment)-1] == '，' or segment[len(segment)-1] == '，' or segment[len(segment)-1] == '，':
                    pass
                else:
                    labels.append(self.schema[''])
                pattern = re.compile('，|。|？')            # 查找'，'，'。'，'？'
                segment = pattern.sub("", segment)         # 删除
                for j in range(int(len(segment)/self.max_length) + 1):
                    sentence = segment[j * self.max_length: (j+1) * self.max_length]
                    if sentence.strip() == "":             # 如果为空字符则跳到下一句话
                        continue
                    self.sentences.append(sentence)
                    input_ids = self.encode_sentence(sentence)     # 调用编码函数，把每句话转为输入索引
                    label = labels[j * self.max_length: (j+1) * self.max_length]
                    label = self.padding(label, -1)                # 用-1填充标签序列
                    assert len(input_ids) == len(label), (len(input_ids), len(label))
                    self.data.append([torch.LongTensor(input_ids), torch.LongTensor(label)])  # [[[输入索引],[标签]],...]
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":                        # 如果字典路径为words.txt
            for word in jieba.cut(text):                                    # 使用结巴对每句话分词
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))  # 把分词结果中每个词在字典中对应的序号添加到input_id
        else:
            for char in text:
                # 字典路径为char.txt，把每句话中的每个字符在字典中对应的序号添加到input_id
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)  # 如果填充输入索引的话，调用填充函数
        return input_id                        # 返回一维列表

        # 补齐或截断输入的序列，使其可以在一个batch内运算

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]                        # 截断
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))  # 用0补齐
        return input_id

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:    # 打开标签文件，把文件内容转为dict
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1        # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = Dataset(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    # dg = DataGenerator("../ner_data/train.txt", Config)
    pass
