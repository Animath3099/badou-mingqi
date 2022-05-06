#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"D:\badou\course\week4 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = 100 #int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    index = 0
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        distance_to_center = calc_distance(vectors[index], kmeans.cluster_centers_[label]) #计算每句话到中心的距离
        sentence_label_dict[label].append([sentence, distance_to_center])         #同标签的放到一起
        index += 1
    #计算每类的类内平均距离
    data = [[label, sent_dist, sum([var[1] for var in sent_dist])/len(sent_dist)] for label, sent_dist in sentence_label_dict.items()]
    #按距离排序
    data = sorted(data, key=lambda x:x[2])
    for label, sentences, avg_dist in data:
        print("cluster %s :" % label, avg_dist)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i][0].replace(" ", ""))
        print("---------")

#计算向量欧式距离
def calc_distance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

if __name__ == "__main__":
    main()