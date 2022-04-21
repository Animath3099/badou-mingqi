from config import Config
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel, BertConfig
"""
建立网络模型结构
"""


class SBERT(nn.Module):
    def __init__(self, config):
        super(SBERT, self).__init__()
        pretrain_model_path = config["pretrain_model_path"]                 # 预训练模型路径
        # self.bert_config = BertConfig.from_pretrained(pretrain_model_path)
        self.bert_encoder = BertModel.from_pretrained(pretrain_model_path, num_hidden_layers=4)  # BERT编码器
        self.pool_type = config["pool_type"]                                # 平均池化
        self.classify_layer = nn.Linear(3 * 768, 2)                         # 分类层
        self.activation = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()                                   # 交叉熵


    def forward(self, sentence1, sentence2=None, target=None):
        # 同时传入两个句子
        if sentence2 is not None:
            # BERT编码器输出的第一个张量为（N,L,768），第二个张量为（N,768）
            if self.pool_type == "cls":
                pool_vector1 = self.bert_encoder(sentence1)[1]
                pool_vector2 = self.bert_encoder(sentence2)[1]
            elif self.pool_type == "mean":
                vector1 = self.bert_encoder(sentence1)[0]
                pool_vector1 = torch.mean(vector1, dim=1)
                vector2 = self.bert_encoder(sentence2)[0]
                pool_vector2 = torch.mean(vector2, dim=1)
            else:
                assert self.pool_type == "max"
                vector1 = self.bert_encoder(sentence1)[0]
                pool_vector1 = torch.max(vector1, dim=1)
                vector2 = self.bert_encoder(sentence2)[0]
                pool_vector2 = torch.max(vector2, dim=1)
            join_vector = torch.cat((pool_vector1, pool_vector2, torch.abs(pool_vector1 - pool_vector2)), dim=1)
            result = self.classify_layer(join_vector)            # (N,2)
            result = self.activation(result)
            # 如果有标签，则计算loss
            if target is not None:
                return self.loss(result, target.squeeze())      # target,(N)
            # 如果无标签，计算余弦距离
            else:
                return self.cosine_distance(pool_vector1, pool_vector2)
        # 单独传入一个句子时，认为正在使用向量化能力
        else:
            # BERT编码器输出的第一个张量为（N,L,768），第二个张量为（N,768）
            if self.pool_type == "cls":
                pool_vector1 = self.bert_encoder(sentence1)[1]
            elif self.pool_type == "mean":
                vector1 = self.bert_encoder(sentence1)[0]
                pool_vector1 = torch.mean(vector1, dim=1)
            else:
                assert self.pool_type == "max"
                vector1 = self.bert_encoder(sentence1)[0]
                pool_vector1 = torch.max(vector1, dim=1)
            return pool_vector1

    # 输入一个句子和一组备选，输出最高分命中的index
    def most_similar(self, input_id, target_ids):
        input_ids = torch.stack([input_id] * len(target_ids))
        res = self.forward(input_ids, target_ids)
        return int(torch.argmax(res))

    # 计算余弦距离
    # 0.5 * (1 + cosine)的目的是把-1到1的余弦值转化到0-1，这样可以直接作为得分
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 0.5 * (1 + cosine)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    model = SBERT(Config)
    # print(model.state_dict().keys())
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    y = model(s1, s2, l)
    print(y)