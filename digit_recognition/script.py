# -*- coding : utf-8 -*-
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "./input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


def get_mnist_data():
    df = pd.read_csv('./input/train.csv')
    X = df.drop('label', axis=1)
    Y = df['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train.values, Y_train.values, X_test.values, Y_test.values


def initialize_weights_and_bias(dim, output_classes_count):
    W = np.random.rand(dim, output_classes_count) * 0.01
    b = np.zeros((1, output_classes_count))
    return W,b


def calculate_gradients(probs, n, Y):
    probs[range(n), Y] -= 1
    probs /= n
    dW = np.dot(X_train.T, probs)
    db = np.sum(probs, axis=0, keepdims=True)
    return dW, db


def optimize(X_train, Y_train, W, b, alpha=0.01, iter_count=2000):
    for i in range(iter_count):
        Z = np.dot(X_train, W) + b 
        exp_z = np.exp(Z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True) # 计算logistic softmax概率
        log_probs = -np.log(probs[range(n), Y_train])
        loss = (1 / n) * np.sum(log_probs) # 计算损失函数

        if i % 100 == 0:
            print("iteration %d: loss %f" % (i, loss))

        dW, db = calculate_gradients(probs, n, Y_train)
        W += -alpha * dW # 梯度下降更新参数
        b += -alpha * db


def calculate_accuracy(X_test, Y_test, W, b): # 验证
    scores = np.dot(X_test, W) + b
    predicted_class = np.argmax(scores, axis=1) # 概率最大的类
    print('training accuracy: %.2f' % (np.mean(predicted_class == Y_test) * 100))


def predict(X, W, b, n): # 测试
    Z = np.dot(X, W) + b
    predictions = np.argmax(Z, axis=1)
    indexes = list(range(1, n+1))
    csv = np.column_stack((np.array(indexes), predictions))
    np.savetxt('pred.csv', csv, fmt='%d', delimiter=',', header=" ImageId, label")


X_train, Y_train, X_test, Y_test = get_mnist_data()
X_train = X_train/255 # 数值归一化
X_test = X_test/255
W, b = initialize_weights_and_bias(X_train.shape[1], 10) # 初始参数
alpha = 0.05 # 学习率
n = X_train.shape[0] # 训练样本数

optimize(X_train, Y_train, W, b, alpha) # 参数优化
calculate_accuracy(X_test, Y_test, W, b) # 模型算法验证评估





