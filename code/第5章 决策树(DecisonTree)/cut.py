import numpy as np
import pandas as pd

from math import log


class DNode:
    def __init__(self, root=None, feature=None, feature_name=None, label=None, data=None):
        """
        有两种节点（root 保存）
        1. 分类节点 root = False（所有属性）：feature、feature_name、tree {}(可能不止两个子节点)
        2. 叶子节点 root = True （所有属性）：label、

        3. data 为叶子节点保存原来数据的属性
        """
            
        
        self.root = root
        self.feature = feature
        self.feature_name = feature_name
        self.label = label
        self.tree = {}
        self.data = data

    def add_node(self, val, node):
        self.tree[val] = node
    
    def copy(self):
        c = {
            'root': self.root,
            'feature': self.feature,
            'feature_name': self.feature_name,
            'label': self.label,
            'tree': self.tree,
            'data': self.data,
        }
        return c

    def sons_merge(self):
        self.root = False
        data = [s.data for s in self.tree.values()]
        data = np.concatenate(data, axis=0)
        self.data = data
        y_train = pd.Series(data[:, -1])
        label = y_train.value_counts().sort_values(ascending=False).index[0]
        self.label = label

    def reduction(self, c):
        self.root = c['root']
        self.feature = c['feature']
        self.feature_name = c['feature_name']
        self.label = c['label']
        self.tree = c['tree']
        self.data = c['data']
        



class DTree:
    def __init__(self, epsilon=0.1):
        self._tree = None
        self.epsilon = epsilon
        self.alpha = 0.1

    @staticmethod
    def cal_ent(dataset):
        data_length = len(dataset)
        labels = {}
        for i in range(data_length):
            label = dataset[i][-1]
            if label not in labels:
                labels[label] = 0
            labels[label] += 1
        ent = -sum([(p/data_length)*log(p/data_length, 2)
                    for p in labels.values()])
        print('ent', ent)

        return ent

    def cond_ent(self, dataset, axis=0):
        # 计算经验条件熵
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
        return ent - c_ent

    def info_gain_train(self, dataset):
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
        print('max_feature max_feature_name', max_feature, max_feature_name)

        if max_info_gain < self.epsilon:
            return DNode(root=True,
                         label=y_train.value_counts().sort_values(ascending=False).index[0], data=np.array(train_data))

        # 构建子集
        node_tree = DNode(root=False,
                          feature_name=max_feature_name, feature=max_feature)

        # 将那些数据按 max_feature_name 分开
        feature_list = train_data[max_feature_name].value_counts().index

        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop(
                [max_feature_name], axis=1)

            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree
    
    def cut(self, alpha):
        self.alpha = alpha if alpha else self.alpha
        min_loss = self.cal_loss()
        def travel(node):
            nonlocal min_loss
            if not node.root:
                # 找到叶子节点的父节点
                son_nodes = list(node.tree.values())
                # print('type son_nodes', type(son_nodes))
                roots = [n.root for n in son_nodes]
                if not False in roots:      
                    # 全部是叶子节点
                    # 删除节点之前，保存节点
                    old_node = node.copy()
                    # 合并节点
                    node.sons_merge()
                    # 计算新的损失函数
                    new_loss = self.cal_loss()
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
                        if_re = travel(node)
                        if if_re == 1:
                            # 此节点是儿子都是叶节点，并且进行了剪枝
                            re = 1
                        elif if_re == 2:
                            # 重新检查当前节点
                            i -= 1
                        i += 1
                    if re:
                        return 2
            return 0
        travel(self.root_node)
    
    def all_leaves(self):
        leaves = []
        root_node = self.root_node
        def travel(node):
            if node.root ==  False:
                for n in node.tree.values():
                    travel(n)
            else:
                leaves.append(node)
        travel(root_node)
        return leaves

    def cal_loss(self):
        leaves = self.all_leaves()
        first = []
        for _l in leaves:
            # print('_l data', _l.data)
            # print(type(_l.data))
            ent = self.cal_ent(_l.data)
            first.append(ent)
        res = sum(first) + self.alpha * len(leaves)
        return res




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


def main():
    datasets, labels = create_data()
    data_df = pd.DataFrame(datasets, columns=labels)
    dt = DTree()
    root_node = dt.fit(data_df)
    dt.root_node = root_node
    ls = dt.all_leaves()
    # 进行减枝
    dt.cut(alpha=0.5)


if __name__ == "__main__":
    main()
 
