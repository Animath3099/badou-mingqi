import torch

from sentence_match_as_sentence_encoder.config import Config
from sentence_match_as_sentence_encoder.loader import DataGenerator
from sentence_match_as_sentence_encoder.model import SiameseNetwork


class QASystem:
    def __init__(self, know_base_path):
        self.data_gen = DataGenerator(data_path=know_base_path, config=Config)
        self.data_gen.load()
        self.model = SiameseNetwork(Config)
        self.model.load_state_dict(
            torch.load(
                "./sentence_match_as_sentence_encoder/model_output/epoch_80.pth"
            )
        )
        self.standard_ques = {index:ques for ques,index in
                              self.data_gen.schema.items()}
        self.knowb_vectors = {
            index: [
                self.model(torch.unsqueeze(tensor, dim=0)) for tensor in values
            ]
            for index, values in self.data_gen.knwb.items()
        }

    def query(self, question):
        qus_vec = self.data_gen.encode_sentence(question)
        ques_tensor = torch.LongTensor(qus_vec).unsqueeze(dim=0)
        model_vec = self.model(ques_tensor)
        max_cos = -1
        max_label = None
        for label, vecs in self.knowb_vectors.items():
            label_max_cos = max(
                self.model.cosine_distance(model_vec, vec) for vec in vecs
            )
            if label_max_cos > max_cos:
                max_cos = label_max_cos
                max_label = label
        return self.standard_ques.get(max_label)


if __name__ == "__main__":
    qas = QASystem(Config["train_data_path"])
    while True:
        question = input("请输入问题：")
        res = qas.query(question)
        print("命中问题：", res)
        print("-----------")
