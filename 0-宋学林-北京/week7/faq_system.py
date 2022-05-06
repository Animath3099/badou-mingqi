import json
import numpy as np
import torch
from config import Config
from model import SiameseNetwork


class QASystem:
    #入参：
    #1.知识库路径
    #2.训练好的模型权重路径
    #3.字表文件路径，需要和训练时一致
    def __init__(self, know_base_path, model_path, vocab_path):
        self.load_model(model_path, vocab_path)
        self.load_know_base(know_base_path)
        print("知识库加载完毕，可以开始问答！")

    #加载知识库
    # 1.将知识库里的每一条问题向量化
    # 2.记录每条问题对应的标准问是哪条
    def load_know_base(self, know_base_path):
        self.know_base_question_vectors = []
        self.index_to_target = {}
        question_index = 0
        with open(know_base_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                content = json.loads(line)
                questions = content["questions"]  #标准问下的所有相似问
                target = content["target"]        #标准问
                for question in questions:
                    #每个问题编号，记录对应的标准问
                    self.index_to_target[question_index] = target
                    question_index += 1
                    #每条问题转化成向量
                    vector = self.string_to_vector(question)
                    self.know_base_question_vectors.append(vector)
        #将list转化为矩阵，方便后续运算
        self.know_base_question_vectors = np.array(self.know_base_question_vectors)
        return

    #加载训练好的模型权重和字表文件
    def load_model(self, model_path, vocab_path):
        #加载字表
        self.vocab = self.load_vocab(vocab_path)
        Config["vocab_size"] = len(self.vocab)
        #加载训练好的模型
        self.model = SiameseNetwork(Config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        return

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        return token_dict

    #文本到向量
    def string_to_vector(self, string):
        #文本按照词表转换成数字标号
        input_id = []
        for char in string:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        #传入这句话给模型，将序号转成tensor，batch_size=1
        vector = self.model(torch.LongTensor([input_id]))
        #向量做归一化，为了后续计算余弦值方便，并转成numpy
        vector = vector.cpu().detach().numpy()
        vector /= np.sqrt(np.sum(np.square(vector)))
        return vector


    #寻找最接近的向量
    def find_most_similar_vector(self, question_vector):
        #question_vector: 1 * 128
        #know_base_question_vectors: n * 128,  n为问题总数量
        #输出similaritys: 1 * n, 每个数是与对应问题的余弦相似度
        similaritys = np.dot(question_vector, self.know_base_question_vectors.T)
        return np.argmax(similaritys)

    def query(self, question):
        # 输入问题转化到向量
        question_vector = self.string_to_vector(question)
        # 计算得到最相似的向量index
        most_similar_vector_index = self.find_most_similar_vector(question_vector)
        # 取出对应的标准问
        target = self.index_to_target[most_similar_vector_index]
        return target

if __name__ == '__main__':
    qas = QASystem("train.json", "epoch_10.pth", "chars.txt")
    while True:
        question = input("请输入问题：")
        res = qas.query(question)
        print("命中问题：", res)
        print("-----------")
