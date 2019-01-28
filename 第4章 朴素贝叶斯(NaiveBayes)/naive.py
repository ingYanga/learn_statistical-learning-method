import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width',
                  'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:, :-1], data[:, -1]


class NaiveBayes:
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x-avg, 2) for x in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    def summarize(self, train):
        s = [(self.mean(i), self.stdev(i)) for i in zip(*train)]
        return s


    def fit(self, X, y):
        """
        这个函数用来得到概率密度函数里面的标准差和平均数
        而且每个标签都要分开来算

        : para X 数组 [[1, 2, 3], [1, 2, 3]]
        : para y 是标签 [0, 1]
        """
        labels = list(set(y))
        data = {label:[] for label in labels}

        # 根据标签分开数据
        for x, label in zip(X, y):
            data[label].append(x)
        # 分类后，计算每个类中的每个特征的平均数和标准差
        self.model = {label:self.summarize(value) for label, value in data.items() }

        return '训练完毕'

    def cal_probabilities(self, input_data):
        """
        计算每个分类的可能性
        """
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(input_data)):
                _mean, _stdev = value[i]
                # 贝叶斯公式分子的左部分
                probabilities[label] *= self.gaussian_probability(input_data[i], _mean, _stdev)
        return probabilities

    def predict(self, input_data):
        label = sorted(self.cal_probabilities(input_data).items(), key=lambda x : x[-1])[-1][0]
        return label

        
def main():
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # print('X, y', X, y)
    # print('X_train, X_test, y_train, y_test', X_train, X_test, y_train, y_test)
    model = NaiveBayes()
    print(model.fit(X_train, y_train))

    print(model.predict([10,  3.2,  1.3,  0.2]))


if __name__ == "__main__":
    main()
