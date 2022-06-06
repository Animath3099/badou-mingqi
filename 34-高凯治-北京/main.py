import time
import torch
import os
import numpy as np
import logging
from config import Config
from model import SBERT, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from plot_loss_curve import plot_loss_acc


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"          # 设置使用哪张显卡

# 创建一个logger并设置日志等级
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)                 #告诉logger要记录哪些级别的日志
# 定义日志文件
logFile = os.path.join(Config["model_path"], 'run_process.log')

# 定义Handler的日志输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s')

# 创建一个FileHandler,并将日志写入指定的日志文件中
fileHandler = logging.FileHandler(filename=logFile, mode='a', encoding='utf-8')   #追加写的方式，'w'覆盖之前的日志
fileHandler.setLevel(logging.INFO)             #告诉Handler要记录哪些级别的日志
fileHandler.setFormatter(formatter)

# 创建一个StreamHandler,将日志输出到控制台
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.INFO)
streamHandler.setFormatter(formatter)

# 添加Handler
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)
"""
模型训练主程序
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)         # 93
    # 加载模型
    model = SBERT(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger, train_data)
    # 训练
    start = time.time()
    train_epoch_loss = []
    eval_acc = []
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id1, input_id2, labels = batch_data                  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id1, input_id2, labels)
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        train_epoch_loss.append(np.mean(train_loss))
        eval_acc.append(evaluator.eval(epoch))
    epoch = range(1, config["epoch"] + 1)
    plot_loss_acc(epoch, train_epoch_loss, eval_acc)
    print(time.time() - start)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    main(Config)