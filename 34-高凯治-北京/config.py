"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "pretrain_model_path": r"D:\八斗Ai名企实战班\pretrain_model\chinese_bert_likes\bert-base-chinese",
    "vocab_path": r"D:\八斗Ai名企实战班\pretrain_model\chinese_bert_likes\bert-base-chinese\vocab.txt",
    "bert_layer_num": 4,
    "max_length": 24,
    "epoch": 100,
    "batch_size": 32,
    "epoch_data_size": 3000,                   # 每轮训练中采样数量
    "positive_sample_rate": 0.5,               # 正样本比例
    "optimizer": "adam",
    "learning_rate": 2e-5,
    "dropout": 0.2,
    "warmup_steps": 100,
    "display_interval": 500,
    "pool_type": "mean",
}