from config import Config
import os
import json
import torch
from model import SBERT
from collections import defaultdict
from loader import load_schema
from transformers import BertTokenizer



def encode_sentence(text):
    vocab_path = Config["vocab_path"]
    tokenizer = BertTokenizer(vocab_path)
    input_id = tokenizer.encode(text, max_length=Config["max_length"], pad_to_max_length=True)
    return input_id

class QASystem:
    def __init__(self, know_base_path, model_path):             # 调用知识库数据初始化问答系统
        self.model = SBERT(Config)
        self.model.load_state_dict(torch.load(model_path), strict=False)           # 调用预训练模型
        self.schema = load_schema(Config["schema_path"])
        self.load_know_base(know_base_path)
        self.knwb_to_vector()                                    # 使用预训练模型编码知识库
        print("模型加载完毕，可以开始问答！")

    def load_know_base(self, know_base_path):
        self.knwb = defaultdict(list)
        self.index_to_target = {v: k for k, v in self.schema.items()}                               # 存放索引到标准问的字典(29)
        with open(know_base_path, encoding="utf8") as f:
            for index, line in enumerate(f):                    # 循环知识库文件中的每一行知识
                line = json.loads(line)                         # 字符串转字典
                assert isinstance(line, dict)
                questions = line["questions"]
                label = line["target"]
                for question in questions:
                    input_id = encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    self.knwb[self.schema[label]].append(input_id)     # self.knwb->{0：[[],[]...]}

    # 将知识库中的问题向量化，为匹配做准备
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        self.knwb_vectors = []
        for standard_question_index, question_ids in self.knwb.items():  # self.knwb->{0：[[],[]...]}
            for question_id in question_ids:
                # 记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)                   # 二维列表[[],[],..],2342个相似问

        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)    # 把二维列表堆叠为二维张量，在dim维度新增一个维度[2342, 24]
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                question_matrixs = question_matrixs.cuda()
            for i in range(2):
                self.knwb_vectors.extend(self.model(question_matrixs[i * 1171: (i + 1) * 1171]))            # 输入（N,L）
            # 将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.stack(self.knwb_vectors, dim=0)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return


    def query(self, question):                       # 请求函数（用户问）
        question_ids = encode_sentence(question)     # 编码用户问
        question_ids = torch.LongTensor(question_ids).unsqueeze(0)
        if torch.cuda.is_available():
            question_ids = question_ids.cuda()
        with torch.no_grad():
            test_question_vector = self.model(question_ids)        # 不输入labels，使用模型当前参数得到用户问的编码结果
            # 通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            # test_question_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
            res = torch.mm(test_question_vector, self.knwb_vectors.T)  # 矩阵乘法
            hit_index = int(torch.argmax(res.squeeze()))                            # 命中问题标号（1，n）->scaler
            hit_index = self.question_index_to_standard_question_index[hit_index]   # 转化成标准问编号
        return self.index_to_target[hit_index]      # 利用索引到目标问字典得到目标问


if __name__ == '__main__':
    model_path = os.path.join(Config["model_path"], "epoch_100.pth")
    qas = QASystem('../data/data.json', model_path)
    while True:
        question = input("请输入问题：")
        res = qas.query(question)
        print("命中问题：", res)
        print("-----------")