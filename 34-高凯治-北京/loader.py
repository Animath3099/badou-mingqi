import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from config import Config
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.tokenizer.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]     # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.max_length = config["max_length"]
        self.data_type = None                                # 用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)                    # 字符串转字典
                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)    # self.knwb->{0：[[],[]...]}
                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])             # self.data->[[[编码后的输入],[标签索引]],...]
        return

    def encode_sentence(self, text):
        input_id = self.tokenizer.encode(text, max_length=self.max_length, pad_to_max_length=True)
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]               # 训练数据长度为超参数
        else:
            assert self.data_type == "test", self.data_type     # 测试数据长度可变
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()                   # 随机生成一个训练样本
        else:
            return self.data[index]

    # 依照一定概率生成负样本或正样本
    # 负样本从随机两个不同的标准问题中各随机选取一个
    # 正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())        # 知识库的所有键（即标准问对应的标签）转为列表
        # 随机正样本
        if random.random() <= self.config["positive_sample_rate"]:
            p = random.choice(standard_question_index)          # 返回一个列表，元组或字符串的随机项
            # 如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
            if len(self.knwb[p]) < 2:
                return self.random_train_sample()
            else:
                s1, s2 = random.sample(self.knwb[p], 2)         # 从列表中随机采样两个元素
                return [s1, s2, torch.LongTensor([1])]          # [[],[],[1]]
        # 随机负样本
        else:
            p, n = random.sample(standard_question_index, 2)     # 随机采样两个标准问
            s1 = random.choice(self.knwb[p])                     # 返回一个列表，元组或字符串的随机项
            s2 = random.choice(self.knwb[n])
            return [s1, s2, torch.LongTensor([0])]              # [[],[],[0]]


# 加载词元化器
def load_vocab(vocab_path):
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer


# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())                             # f.read()读整个文件，返回字符串类型


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle, drop_last=True)
    return dl


if __name__ == "__main__":
    dg = DataGenerator(Config["train_data_path"], Config)
    print(dg[0])