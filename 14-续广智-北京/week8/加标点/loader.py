# -*- coding: utf-8 -*-

import json
import re
#import os
#import torch
#import random
#import jieba
import numpy as np
from torch.utils.data import DataLoader

"""
数据加载
"""

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = self.load_schema(config["schema_path"])
        self.config["class_num"] = len(self.schema)
        self.max_length = config["max_length"]
        self.data = self.load()

    def load(self):
        # load data
        with open(self.path, 'r', encoding='utf-8') as fin:
            text_list = fin.readlines()

        # pattern: g1: chars g2: any positive number of punctuations (，。？）
        pattern = re.compile(r'(.*?)([%s]+)'\
                %(''.join([vv for vv in self.schema.keys() if len(vv)>0])))
        res = []
        for tii in text_list:
            tii = tii.strip()
            labelii = []
            dataii = ''
            for mjj in pattern.finditer(tii):
                labelii.extend([self.schema['']] * (len(mjj.group(1)) - 1)) # labels of g1 texts
                labelii.append(self.schema[mjj.group(2)[-1]])  # label of last punctuation in g2
                dataii += mjj.group(1) # g1 texts
                assert len(dataii) == len(labelii), 'WTF?'
            labelii = np.array(labelii)
            res.append((dataii, labelii))

        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("./data/train_corpus.txt", Config)

    for ii in range(3):
        print('########### Sample %d ##############' %ii)
        print(dg.data[ii][0][:30])
        print(dg.data[ii][1][:30])
        print()

