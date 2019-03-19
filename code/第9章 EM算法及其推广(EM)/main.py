import numpy as np
import random
import math
import time


def create_data(mu0, sigma0, mu1, sigma1, alpha0, alpha1):
    '''
    初始化数据集
    这里通过服从高斯分布的随机函数来伪造数据集
    :param mu0: 高斯0的均值
    :param sigma0: 高斯0的方差
    :param mu1: 高斯1的均值
    :param sigma1: 高斯1的方差
    :param alpha0: 高斯0的系数
    :param alpha1: 高斯1的系数
    :return: 混合了两个高斯分布的数据
    '''
    #定义数据集长度为1000
    length = 1000
    
    #初始化第一个高斯分布，生成数据，数据长度为length * alpha系数，以此来
    #满足alpha的作用
    data0 = np.random.normal(mu0, sigma0, int(length * alpha0))
    #第二个高斯分布的数据
    data1 = np.random.normal(mu1, sigma1, int(length * alpha1))

    #初始化总数据集
    #两个高斯分布的数据混合后会放在该数据集中返回
    dataSet = []
    #将第一个数据集的内容添加进去
    dataSet.extend(data0)
    #添加第二个数据集的数据
    dataSet.extend(data1)
    # 对总的数据集进行打乱（其实不打乱也没事，只不过打乱一下直观上让人感觉已经混合了
    # 读者可以将下面这句话屏蔽以后看看效果是否有差别）
    random.shuffle(dataSet)

    #返回伪造好的数据集
    return dataSet


def cal_gauss(dataSetArr, mu, sigmod):
    '''
    根据高斯密度函数计算值
    依据：“9.3.1 高斯混合模型” 式9.25
    注：在公式中y是一个实数，但是在EM算法中(见算法9.2的E步)，需要对每个j
    都求一次yjk，在本实例中有1000个可观测数据，因此需要计算1000次。考虑到
    在E步时进行1000次高斯计算，程序上比较不简洁，因此这里的y是向量，在numpy
    的exp中如果exp内部值为向量，则对向量中每个值进行exp，输出仍是向量的形式。
    所以使用向量的形式1次计算即可将所有计算结果得出，程序上较为简洁
    :param dataSetArr: 可观测数据集
    :param mu: 均值
    :param sigmod: 方差
    :return: 整个可观测数据集的高斯分布密度（向量形式）
    '''
    #计算过程就是依据式9.25写的，没有别的花样
    result = (1 / (math.sqrt(2 * math.pi) * sigmod**2)) * \
        np.exp(-1 * (dataSetArr - mu) * (dataSetArr - mu) / (2 * sigmod**2))
    #返回结果
    return result

class EM_Gauss:
    def load_data(self, data):
        # 保存所有数据
        self.data = np.array(data).reshape(1, len(data))

    def init_args(self, all_args):
        # 保存所有参数
        self.all_args = all_args
        # 模型数目
        self.K = len(all_args) // 3
        # 样本长度
        _, self.N = self.data.shape

    # E 步骤
    def E(self):
        # 用来保存中间值
        t_d = {}
        # 保存分母
        _sum = np.zeros((1, self.N))
        # print('_sum', _sum.shape)
        for k in range(self.K):
            str_k = str(k)
            key = 'gauss' + str_k
            t_d[key] = cal_gauss(self.data, self.all_args["mu"+str_k], self.all_args['sigma'+str_k])
            t_d[key] = self.all_args["alpha"+str_k] * t_d[key]
            _sum += t_d[key]

        all_gamma = {}
        for k in range(self.K):
            str_k = str(k)
            key1 = 'gamma' + str_k
            key2 = 'gauss' + str_k
            all_gamma[key1] = t_d[key2] / _sum
        
        return all_gamma


    # 步骤 M 更新参数
    def M(self, all_gamma):
        """
        更新模型参数
        """
        for k in range(self.K):
            str_k = str(k)
            sum_gamma = np.sum(all_gamma['gamma'+str_k])

            self.all_args['mu'+str_k] = np.sum(all_gamma['gamma'+str_k] * self.data) / sum_gamma
            self.all_args['sigma'+str_k] = np.sqrt(
                np.sum(all_gamma['gamma'+str_k] * np.square(self.data - self.all_args['mu'+str_k])) / sum_gamma
            )
            self.all_args['alpha' + str_k] = sum_gamma / self.N


    def train(self, max_iters=100):
        

        for t in range(max_iters):
            print('t', t)
            # 计算 E 步骤
            all_gamma = self.E()

            # 计算 M 步骤
            self.M(all_gamma)

        return self.all_args

def main():
    # 加载自己伪造的高斯数据
    # 假设伪造的数据只有两个高斯混合
    # mu0 sigma0 alpha0 是第一个高斯
    # mu1 sigma1 alpha1 是第二个高斯
    # alpha0 alpha1 是比例系数 相加为 1
    mu0, sigma0, alpha0 = -2.0, 0.5, 0.3
    mu1, sigma1, alpha1 = 0.5, 1.0, 0.7 
    data = create_data(mu0, sigma0, mu1, sigma1, alpha0, alpha1)

    # 创建 EM_Gauss 类
    em = EM_Gauss()
    # 加载数据
    em.load_data(data)
    # 初始化参数
    # 在 M 步中有三个参数要更新
    # 现在定义每个参数的初始值

    all_args = {
        'mu0': 0.,
        'sigma0': 1.0,
        'alpha0': 0.5,
        'mu1': 1.0,
        'sigma1': 1.0,
        'alpha1': 0.5 
    }
    em.init_args(all_args)
    all_args = em.train(max_iters=500)

    print(all_args)

if __name__ == "__main__":
    main()
