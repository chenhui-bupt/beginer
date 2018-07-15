# -*- coding: utf-8 -*-

import sys
import os
from collections import defaultdict
import math

class MaxEnt(object):
    def __init__(self):
        self.feats = defaultdict(int)
        self.trainset = []  # 训练集
        self.labels = set()  # 标签

    def load_data(self, file):
        for line in open(file):
            fields = line.strip().split()
            if len(fields) < 2: continue   # 特征数要大于两列
            label = fields[0]
            self.labels.add(label)
            for f in set(fields[1:]):
                self.feats[(label, f)] += 1  # (label, f)元祖是特征
            self.trainset.append(fields)
        print(self.feats)

    def _initparams(self):  #初始化参数
        self.size = len(self.trainset)
        self.M = max([len(record) - 1 for record in self.trainset])  # GIS的M参数
        self.ep_ = [0.0] * len(self.feats)
        for i, f in enumerate(self.feats):
            self.ep_[i] = float(self.feats[f])/float(self.size)  # 计算经验分布的特征期望
            self.feats[f] = i  # 为每一个特征函数分配id
        self.w = [0.0] * len(self.feats)  # 初始化权重
        self.lastw = self.w

    def probwgt(self, features, label):  # 计算每个特征权重的指数
        wgt = 0.0
        for f in features:
            if (label, f) in self.feats:
                wgt += self.w[self.feats[(label, f)]]  # 权重
        return math.exp(wgt)

    """
    calculate feature expectation on model distribution
    """
    def Ep(self):  # 特征函数
        ep = [0.0] * len(self.feats)
        for record in self.trainset:  # 从训练集迭代输出特征
            features = record[1:]
            prob = self.calprob(features)
            for f in features:
                for w, l in prob:
                    if(l, f) in self.feats:  # 来自训练数据的特征
                        idx = self.feats[(l, f)]  # 获取特征id
                        ep[idx] += w * (1.0/self.size)  # sum(1/N * f(y,x) * p(y|x)), p(x) = 1/N
        return ep

    def _convergence(self, lastw, w):
        for w1, w2 in zip(lastw, w):
            if abs(w1 - w2) >= 0.01:
                return False
        return True

    def train(self, max_iter=1000):
        self._initparams()
        for i in range(max_iter):
            print("iter %d ..." % (i + 1))
            self.ep = self.Ep()  # 计算模型分布的特征期望
            self.lastw = self.w[:]  # 深拷贝
            for i, win in enumerate(self.w):
                delta = 1.0/self.M * math.log(self.ep_[i]/self.ep[i])
                self.w[i] += delta  # 更新w
            print(self.w, self.feats)
            if self._convergence(self.lastw, self.w):
                break

    def calprob(self, features):  # 计算条件概率
        wgts = [(self.probwgt(features, l), l) for l in self.labels]
        Z = sum([w for w,l in wgts])
        prob = [(w/Z, l) for w,l in wgts]
        return prob

    def predict(self, input):
        features = input.strip().split()
        prob = self.calprob(features)
        prob.sort(reverse=True)
        return prob

model = MaxEnt()
model.load_data('maxent_data.txt')
model.train()
print(model.predict("Rainy Happy Dry"))  # 预测结果
print(model.M)





