import numpy as np
import pandas as pd

from math import log


# 创建数据
def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '一般', '是'],
                ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels


class DNode:
    def __init__(self, root=None, feature_name=None, label=None, data=None):
        """
        :param root root=False 是分类节点 root=True 是叶节点
        :param feature_name feature_name 表示分类节点的分类名字
        :param label 根节点的类别，如果还有类别不唯一，通常是分类到最后，最多的那个类别，如果不是叶节点
        label 为 None
        """
        self.root = root
        self.feature_name = feature_name
        self.label = label
        self.tree = {}
        self.data = data
#         print('data', data)

    def add_node(self, val, node):
        # 添加子节点
        self.tree[val] = node

    def copy(self):
        """
        决策树剪枝有个操作叫做预剪枝
        剪枝以后检查如果不需要剪枝就还原
        所以要把这个预剪枝的节点保留下来
        """
        c = {
            'root': self.root,
            'feature_name': self.feature_name,
            'label': self.label,
            'tree': self.tree,
            'data': self.data,
        }
        return c

    def sons_merge(self):
        """
        合并子节点
        
        剪枝是减去叶节点上面的节点
        也就是说，这个分类节点会变成叶节点
        之前被分类的子节点都会被合并
        """
        print('合并子节点')
        self.root = True
        data = [s.data for s in self.tree.values()]
        data = np.concatenate(data, axis=0)
        self.data = data
        y_train = pd.Series(data[:, -1])
        label = y_train.value_counts().sort_values(ascending=False).index[0]
        self.label = label

    def reduction(self, c):
        # 节点的还原
        self.root = c['root']
        self.feature_name = c['feature_name']
        self.label = c['label']
        self.tree = c['tree']
        self.data = c['data']


class DTree:
    def __init__(self, epsilon=0.1, alpha=0.1):
        """
        :param epsilon 信息增益的阈值
        如果最大信息增益小于阈值，就不继续分节点了
        :param alpha 公式 5.14 中的alpha
        """
        self.tree = None
        self.epsilon = epsilon
        self.alpha = alpha

    @staticmethod
    def cal_ent(dataset):
        """
        计算当前数据的熵值 公式 5.1
        
        :param dataset 每行是一个样本，每行最后一个数是样本的类别
        """
        data_length = len(dataset)
        labels = {}
        for i in range(data_length):
            label = dataset[i][-1]
            if label not in labels:
                labels[label] = 0
            labels[label] += 1
        ent = -sum([(p/data_length)*log(p/data_length, 2)
                    for p in labels.values()])

        return ent

    def cond_ent(self, dataset, axis=0):
        """
        计算条件熵 书上公式 5.5
        
        按照某一条件分类，计算每一个分类数据的熵
        计算每一个类别占未分类之前的比例
        将对应频率与熵相乘，最后求和
        
        :param axis 是类别
        """
        data_length = len(dataset)
        cla_via_feature = {}
        for r in range(data_length):
            f = dataset[r][axis]
            if f not in cla_via_feature:
                cla_via_feature[f] = []
            cla_via_feature[f].append(dataset[r])
        ent = sum([(len(p)/data_length)*self.cal_ent(p)
                   for p in cla_via_feature.values()])
        return ent

    def info_gain(self, ent, c_ent):
        """
        信息增益
        """
        return ent - c_ent

    def info_gain_train(self, dataset):
        """
        遍历所有数据，找到最大的信息增益是以哪个类别分类的
        """
        count = len(dataset[0]) - 1
        # 计算熵
        ent = self.cal_ent(dataset)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(dataset, axis=c))
            best_feature.append((c, c_info_gain))
        best_ = max(best_feature, key=lambda x: x[-1])

        return best_

    def train(self, train_data):
        """
        递归函数
        1. 先得到数据
            数据格式是 pandas
        2. 判断是否是叶节点
            1. 所有分类的样本的标签是一样的
            2. 虽然有的标签不一样，但是特征值用完了，现在强行弄成标签一样，类别最多的那个
        3. 计算经验熵和条件经验熵，计算信息增益，算出信息增益最大的那个
            1. 如果最大信息增益小于阈值，强行弄成标签一样，类别最多的那个
        4. 把由 3 产生的数据重复执行前三步
        """
        y_train, features = train_data.iloc[:, -1], train_data.columns[:-1]
        # 如果是都是一样的类，返回叶节点
        if len(y_train.value_counts()) == 1:
            return DNode(root=True, label=y_train.iloc[0], data=np.array(train_data))

        # 没有可分的特征值了，返回
        if len(features) == 0:
            return DNode(root=True,
                         label=y_train.value_counts().sort_values(ascending=False).index[0], data=np.array(train_data))

        # 计算经验熵和经验条件熵
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        if max_info_gain < self.epsilon:
            return DNode(root=True,
                         label=y_train.value_counts().sort_values(ascending=False).index[0], data=np.array(train_data))

        # 构建子集
        node_tree = DNode(root=False,
                          feature_name=max_feature_name)

        # 将那些数据按 max_feature_name 分开
        feature_list = train_data[max_feature_name].value_counts().index

        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f]
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        return node_tree

    def fit(self, train_data):
        self.tree = self.train(train_data)
        print('train done!')
        return self.tree

    def cut(self, alpha):
        print('尝试剪枝')
        self.alpha = alpha if alpha else self.alpha
        min_loss = self.cal_loss()
        print('当前损失', min_loss)

        def travel(node):
            # 在当前作用域中使用上面定义的 min_loss
            nonlocal min_loss
            if not node.root:
                # 找到叶子节点的父节点
                son_nodes = list(node.tree.values())
                roots = [n.root for n in son_nodes]
                if not False in roots:
                    print('找到全是叶节点的分类结点')
                    # 全部是叶子节点
                    # 删除节点之前，保存节点
                    old_node = node.copy()
                    # 合并节点
                    node.sons_merge()
                    # 计算新的损失函数
                    new_loss = self.cal_loss()
                    print('剪枝后的损失', new_loss)
                    if new_loss < min_loss:
                        # 1 代表剪枝
                        print('剪枝')
                        min_loss = new_loss
                        return 1
                    else:
                        node.reduction(old_node)
                else:
                    # 子节点不全是叶节点
                    i = 0
                    re = 0
                    while i < len(son_nodes):
                        node = son_nodes[i]
                        # travel 一共有三种可能的结果
                        # 0 没有剪枝 最简单，什么都不动
                        # 1 当前节点的一个节点剪枝了，那么当前节点有可能变成一个都是叶节点的分类结点
                        # 2 当前节点收到了一个 2。告诉当前节点，你有一个子节点的子节点剪枝了
                        # 你的那个子节点可能变成了一个全是叶节点的分类结点，所以你要重新检查这个子节点
                        if_re = travel(node)
                        if if_re == 1:
                            # 此节点是儿子都是叶节点，并且进行了剪枝
                            re = 1
                        elif if_re == 2:
                            # 重新检查当前子节点
                            i -= 1
                        i += 1
                    if re:
                        return 2
            return 0
        travel(self.tree)

    def all_leaves(self):
        """
        得到所有的叶子节点
        """
        leaves = []
        tree = self.tree

        def travel(node):
            if node.root == False:
                for n in node.tree.values():
                    travel(n)
            else:
                leaves.append(node)
        travel(tree)

        return leaves

    def cal_loss(self):
        """
        计算损失书上公式 5.13
        """
        leaves = self.all_leaves()
        first = []
        for _l in leaves:
            ent = self.cal_ent(_l.data)
            first.append(ent)
        res = sum(first) + self.alpha * len(leaves)
        return res

    def predict(self, X, labels):
        """
        :param X 待预测的数据
        :labels 每个带预测数据的含义
        """
        n = self.tree
        while(not n.root):
            feature_name = n.feature_name
            idx = labels.index(feature_name)
            cls = X[idx]
            n = n.tree[cls]

        # n.root 现在是 False
        # 输出 n.label 就是 类别
        print('predict class ', n.label)


def main():
    datasets, labels = create_data()
    data_df = pd.DataFrame(datasets, columns=labels)
    dt = DTree()
    dt.fit(data_df)
    # 进行减枝
    dt.cut(alpha=0.1)

    p_x = ['青年', '否', '否', '一般', '否']
    dt.predict(p_x, labels)


if __name__ == "__main__":
    main()
