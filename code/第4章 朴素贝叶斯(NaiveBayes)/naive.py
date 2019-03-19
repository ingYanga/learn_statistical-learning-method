import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 创建数据
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width',
                  'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    return data[:, :-1], data[:, -1]

# 利用高斯分布估计概率
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

    # 概率密度函数 用概率密度代替概率
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    
    def summarize(self, train):
        """
        :param train [[1, 2, 3], [4, 5, 6]]
        """
        # *train 可以得到所有元素，按照上面的例子 print(*train)
        # 得到 [1, 2, 3] [4, 5, 6]
        # 所以 zip(*train) 
        # 得到 [1, 4]
        # 所以 i 就是每一个特征的集合
        # 然后计算平均和标准差
        s = [(self.mean(i), self.stdev(i)) for i in zip(*train)]
        return s

    def fit(self, X, y):
        """
        :param X 数组 [[1, 2, 3], [1, 2, 3]]
        :param y 是标签 [0, 1]
        """
        labels = list(set(y))
        data = {label: [] for label in labels}

        # 根据标签分开数据
        for x, label in zip(X, y):
            data[label].append(x)
        # 分类后，计算每个类中的每个特征的平均数和标准差
        self.model = {label: self.summarize(value)
                      for label, value in data.items()}
        
        # 计算 py。并把 py 放在数组最后面
        for label, value in data.items():
            self.model[label].append(len(value)/len(X))
        
        print('train done!')

    def cal_probabilities(self, input_data):
        """
        计算每个类别的可能性
        """
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            # 贝叶斯公式右边一项 p(y)
            py = value[-1]
            for i in range(len(input_data)):
                _mean, _stdev = value[i]
                # 贝叶斯公式 p(y|x)
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], _mean, _stdev)
            probabilities[label] *= py
        return probabilities

    def predict(self, input_data):
        """
        cal_probabilities(input_data)
        返回一个字典
        items()
        返回一个 tupple 
        用 tupple 最后一项排序
        得到排序后的序列
        选最后一个概率最大的
        返回 label
        """
        label = sorted(self.cal_probabilities(
            input_data).items(), key=lambda x: x[-1])[-1][0]
        return label
    
    def score(self, X, y):
        r = 0.0
        for x, real_y in zip(X, y):
            predict_y = self.predict(x)
            if predict_y == real_y:
                r += 1
                
        return r/len(y)

def main():
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = NaiveBayes()
    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))


if __name__ == "__main__":
    main()
