import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC


from DrawTools import DrawTools


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width',
                  'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]



class SVM(object):
    def __init__(self):
        pass

    def load_data(self, X, y):
        self.X = X
        self.y = y

    def kernel(self, x1, x2):
        # print('x1 self.features_len', len(x1), self.features_len)
        if self.kernel_type == 'linear':
            return sum([x1[k]*x2[k] for k in range(self.features_len)])
        elif self.kernel_type == 'poly':
            return (sum([x1[k]*x2[k] for k in range(self.features_len)]) + 1)**2

        return 0

    def g(self, idx):
        r = self.b
        for j in range(self.samples_len):
            r += self.alpha[j]*self.y[j]*self.kernel(self.X[idx], self.X[j])
        return r


    def compute_E(self, idx):
        # print('idx', idx)
        return self.g(idx) - self.y[idx]

    def KKT(self, idx):
        t = self.y[idx]*self.g(idx)
        if self.alpha[idx] == 0:
            return t >= 1
        elif 0 < self.alpha[idx] < self.C:
            return t == 1
        else:
            return t <= 1

    def init_args(self, kernel_type='linear', c=0.0001):
        self.samples_len, self.features_len = self.X.shape
        self.b = 0.0
        self.kernel_type = kernel_type 
        self.alpha = np.zeros(self.samples_len)
        self.E = [self.compute_E(i) for i in range(self.samples_len)]
        # 松弛变量
        self.C = c

    def init_alpha(self):
        # 寻找所有属于 (0, C) 变量的点
        l1 = [i for i in range(self.samples_len) if 0 < self.alpha[i] < self.C]
        # 否则遍历所有点
        l2 = [i for i in range(self.samples_len) if i not in l1]

        l1.extend(l2)

        for i in l1:
            # 书上说，要选择最严重的。这里图简单就没有用最严重违反 KKT 条件的
            if self.KKT(i):
                continue
            
            E = self.E[i]
            if E > 0:
                j = min(range(self.samples_len), key=lambda x: self.E[x])
            else:
                j = max(range(self.samples_len), key=lambda x: self.E[x])

            return i, j
        
    def compare(self, alpha, L, H):
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha




    def train(self, max_iters=100):
        for t in range(max_iters):
            i1, i2 = self.init_alpha()
            print("i1, i2", i1, i2)
            # 计算边界 由 old 值计算
            if self.y[i1] == self.y[i2]:
                L = max(0, self.alpha[i1]+self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1]+self.alpha[i2])
            else:
                L = max(0, self.alpha[i2]-self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2]-self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]

            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(
                self.X[i2], self.X[i2]) - 2*self.kernel(self.X[i1], self.X[i2])

            if eta <= 0:
                continue

            # 计算新的 alpha2
            alpha2_new_unc = self.alpha[i2] + self.y[i2] * (E1 - E2) / eta
            # 满足 KKT 条件
            alpha2_new = self.compare(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.y[i1] * \
                self.y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.y[i1] * self.kernel(self.X[i1], self.X[i1]) * (
                alpha1_new-self.alpha[i1]) - self.y[i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new-self.alpha[i2]) + self.b
            b2_new = -E2 - self.y[i1] * self.kernel(self.X[i1], self.X[i2]) * (
                alpha1_new-self.alpha[i1]) - self.y[i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new-self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self.compute_E(i1)
            self.E[i2] = self.compute_E(i2)

        return 'train done!'

    def predict(self, x):
        r = self.b
        for i in range(self.samples_len):
            r += self.alpha[i] * self.y[i] * self.kernel(x, self.X[i])

        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        r = 0
        for i in range(len(X_test)):
            x = X_test[i]
            result = self.predict(x)
            if result == y_test[i]:
                r += 1
        return r / len(X_test)

    def weight(self):
        
        yx = self.y.reshape(-1, 1)*self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w

            
        



def main():
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    plt.scatter(X[:50, 0], X[:50, 1], label='0')
    plt.scatter(X[50:, 0], X[50:, 1], label='1')
    plt.legend()

    svm = SVM()
    svm.load_data(X_train, y_train)
    svm.init_args()
    svm.train(200)

    score = svm.score(X_test, y_test)

    print('score', score)
    a1, a2 = svm.weight()
    b = svm.b
    x_min = min(svm.X, key=lambda x: x[0])[0]
    x_max = max(svm.X, key=lambda x: x[0])[0]

    y1, y2 = (-b - a1 * x_min)/a2, (-b - a1 * x_max)/a2
    plt.plot([x_min, x_max], [y1, y2])
    # print('weight', )
    plt.show()


    # plt.show()

def nb():
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    r = 0
    for i in range(len(X_train)):
        x = X_train[i]
        # x = x.reshape(-1, 1)
        x = np.matrix(x)
        result = clf.predict(x)
        if result[0] == y_train[i]:
            print('result, y_train', result, y_train[i])

            r += 1

    print(r / len(X_train))
    


if __name__ == "__main__":
    main()
    # nb()
