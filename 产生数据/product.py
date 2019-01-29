from sklearn.datasets import load_iris
import numpy as np
import pandas as pd


class GneraterData:
    def __init__(self):
        pass

    def features_with_label(self, features_num=4, draw=False, sample_num=50, labels=1):
        """
        features_num 有几个特征数据 范围是 1～4
        draw bool 是否画出来
        sample_num 每个类别的数量，范围是 1~50
        labels 是有几个样本 范围是 1~3
        """
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['label'] = iris.target
        df.columns = ['sepal length', 'sepal width',
                      'petal length', 'petal width', 'label']
        data = np.array(df.iloc[:, :])
        X = None
        y = None
        for i in range(labels):
            idx = i * 50
            _x = data[idx:idx + sample_num, :features_num]
            _y = data[idx:idx+sample_num, -1].reshape(1, sample_num)
            # print(_y)
            if X is None:
                X = _x
                y = _y
            else:
                X = np.concatenate((X, _x), axis=0)
                y = np.concatenate((y, _y), axis=1)

        return X, y


def main():
    gd = GneraterData()
    X, y = gd.features_with_label(features_num=4, sample_num=10, labels=2)
    # print(y)
    # print(X)
    # print(X.shape, y.shape)


def test():
    gd = GneraterData()
    X, y = gd.features_with_label(features_num=4, sample_num=10, labels=2)
    # print('X.shape y.shape', X.shape, y.shape)
    assert X.shape == (20, 4) and y.shape == (1, 20)
    X1, y1 = gd.features_with_label(features_num=2, sample_num=20, labels=1)
    # print(X1.shape, y1.shape)
    assert X1.shape == (20, 2) and y1.shape == (1, 20)

if __name__ == "__main__":
    # main()
    test()
