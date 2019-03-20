import numpy as np

# 定义 HMM 类
class HMM:
    def __init__(self):
        self.Q = []
        self.V = []
        self.A = []
        self.B = []
        self.O = []
        self.pi = []
        

    def init_args(self, paras):
        """
        :param paras
        paras = {
            'Q': [1, 2, 3],                
            'V': ['红', '白'],
            'A': [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]],
            'B': [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]],
            'O': ['红', '白', '红', '白'],  # 习题10.1的例子
            'pi': [[0.2, 0.4, 0.4]],
        }
        Q 是有几个隐藏状态
        O 是已观测状态的序列
        V 是观测状态的集合
        A 是状态转移概率矩阵
        B 是观测概率矩阵
        pi 是初始状态概率
        """
        # 遍历字典
        for key, value in paras.items():
            setattr(self, key, value)
    
    # 重新设置参数
    def reset_args(self, paras):
        self.init_args(paras)

    def forward(self):
        # 结果保存，最后求和
        result = [0.0] * len(self.Q)
        # 用来保存中间值
        last_result = None
        for t in range(len(self.O)):
            # 找到当前观测值的索引
            v = self.O[t]
            idx = self.V.index(v)
            # 开始前向算法
            for i in range(len(self.Q)):
                # 初值
                if t == 0:
                    # 公式 10.15
                    # print('idx, i', idx, i, self.B[0][0], self.pi[i])
                    result[i] = self.pi[i] * self.B[i][idx]
                    print(f'初值 alpha({i}) {result[i]}')
                else:
                    # 递推公式 10.16
                    # 计算公式中大括号那一部分
                    _sum = sum([ item * self.A[j][i] for j, item in enumerate(last_result)])
                    result[i] = _sum * self.B[i][idx]
                    print(f'alpha({i}) {result[i]}')

            
            # 保存这一层的递推
            last_result = result.copy()

        return sum(result)

def main():
    paras = {
        'Q': [1, 2, 3],
        'V': ['红', '白'],
        'A': [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]],
        'B': [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]],
        'O': ['红', '白', '红'],  # 习题10.2的例子
        'pi': [0.2, 0.4, 0.4],
    }

    hmm = HMM()
    hmm.init_args(paras)
    print('sum', hmm.forward())






if __name__ == "__main__":
    main()
