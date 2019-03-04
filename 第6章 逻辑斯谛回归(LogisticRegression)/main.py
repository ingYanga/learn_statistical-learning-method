import math
from copy import deepcopy


class MaxEntropy():
    def __init__(self, EPS=0.005):
        self.EPS = EPS
        self.Y = set()
        self.samples = {}
        self.num_XY = {}
        self.XY2w = {}
        self.w2XY = {}
        self.n = 0              # 记录特征的数量
        self.EP = {}            # 某一个键值对期望
        self.N = 0              # 样本数量
        self.w = None           # 记录当前参数
        self.last_w = None      # 记录上次参数值，用来判断是否收敛

    def load_data(self, data):
        self.samples = deepcopy(data)
        self.N = len(self.samples)
        # 遍历整个样本
        for _l in self.samples:
            Y = _l[0]
            X = _l[1:]
            self.Y.add(Y)
            for x in X:
                if (x, Y) in self.num_XY:
                    self.num_XY[(x, Y)] += 1
                else:
                    self.num_XY[(x, Y)] = 1
        # 遍历结束后
        self.n = len(self.num_XY)
        self.w = [0] * self.n
        self.last_w = [0] * self.n
        self.EP = [0] * self.n
        self.M = max([len(sample) for sample in self.samples])
        idx = 0
        # 计算期望
        for key, value in self.num_XY.items():
            self.XY2w[key] = idx
            self.w2XY[idx] = key
            self.EP[idx] = value / self.N
            idx += 1

    def compute_Zx(self, X):
        se = 0
        for y in self.Y:
            sw = 0
            for x in X:
                if (x, y) in self.num_XY:
                    idx = self.XY2w[(x, y)]
                    w = self.w[idx]
                    sw += w
            se += math.exp(sw)

        return se
        

    def compute_pyx(self, y, X):
        zx = self.compute_Zx(X)
        _a = []
        for x in X:
            if (x, y) in self.num_XY:
                idx = self.XY2w[(x, y)]
                w = self.w[idx]
                _a.append(w)
            t = math.exp(sum(_a))
        
        return t/zx

    def convergence(self):  # 判断是否全部收敛
        for last, now in zip(self.lastw, self.w):
            if abs(last - now) >= self.EPS:
                return False
        return True

    def predict(self, X):   # 计算预测概率
        Z = self.compute_Zx(X)
        result = {}
        for y in self.Y:
            ss = 0
            for x in X:
                if (x, y) in self.num_XY:
                    ss += self.w[self.XY2w[(x, y)]]
            pyx = math.exp(ss)/Z
            result[y] = pyx
        return result
    
    def compute_ep(self, idx):
        x, y = self.w2XY[idx]
        ep = 0
        for sample in self.samples:
            # P(y|x) 是一个 y 对应多个 x
            # 如果 x 不存在这个样本中。特征函数就会为 0。所以就 continue
            if x not in sample:
                continue
            pyx = self.compute_pyx(y, sample)
            ep += pyx
        ep /= self.N
        return ep

    def train(self, iterations=5000):
        for j in range(iterations):
            self.lastw = self.w[:]
            for i in range(self.n):
                ep = self.EP[i]
                # 计算模型期望
                _ep = self.compute_ep(i)
                self.w[i] += self.M * math.log(ep/_ep)
            if self.convergence():
                return
        



dataset = [['no', 'sunny', 'hot', 'high', 'FALSE'],
           ['no', 'sunny', 'hot', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'high', 'FALSE'],
           ['yes', 'rainy', 'mild', 'high', 'FALSE'],
           ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
           ['no', 'rainy', 'cool', 'normal', 'TRUE'],
           ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
           ['no', 'sunny', 'mild', 'high', 'FALSE'],
           ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
           ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
           ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
           ['yes', 'overcast', 'mild', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
           ['no', 'rainy', 'mild', 'high', 'TRUE']]


def main():
    maxent = MaxEntropy()
    x = ['overcast', 'mild', 'high', 'FALSE']
    maxent.load_data(dataset)
    maxent.train()
    # print('maxent', maxent.n)
    print('predict:', maxent.predict(x))




if __name__ == "__main__":
    main()


