import pandas as pd
from matplotlib import pyplot as plt
from math import log
# 用以画图时正常显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']='sans-serif'

'''
此次作业难度好大，写起来生无可恋
以往实现二叉树都用的C，用Python还是头一回
花了好长一段时间才弄懂如何在Python下实现二叉树
然后算法写起来感觉好难
于是在网上搜了一下，发现都是用list做数据类型
我平常接触dataframe比较多，觉得dataframe写起来更顺手
也更符合数据科学界的用法（毕竟Kaggle上的数据都是csv或者excel，导入肯定用的是pandas）
于是就顺手根据网上代码自己边写边改成了一个dataframe版本的决策树模型
实现过程还参阅了github上sklearn实现CART算法的源码
'''
# 计算给定数据集的信息熵
def compute_Ent(dataset):
    n = len(dataset)
    label_counts = {}
    for item in dataset:
        label_current = item
        if label_current not in label_counts.keys():
            label_counts[label_current] = 0
        label_counts[label_current] += 1
    ent = 0.0
    for key in label_counts:
        prob = label_counts[key] / n
        ent -= prob * log(prob,2)
    return ent
# 按照权重计算各分支的信息熵
def sum_weight(grouped,total_len):
    weight = len(grouped) / total_len
    return weight * compute_Ent(grouped.iloc[:,-1])

# 根据公式计算信息增益
def Gain(column, data):
    lenth = len(data)
    ent_sum = data.groupby(column).apply(lambda x:sum_weight(x,lenth)).sum()
    ent_D = compute_Ent(data.iloc[:,-1])
    return ent_D - ent_sum

# 计算获取最大的信息增益的feature，输入data是一个dataframe，返回是一个字符串
def get_max_gain(data):
    max_gain = 0
    cols = data.columns[:-1]
    for col in cols:
        gain = Gain(col,data)
        if gain > max_gain:
            max_gain = gain
            max_label = col
    return max_label

# 获取data中最多的类别作为节点分类，输入一个series，返回第一个最小值的索引，为字符串
def get_most_label(label_list):
    return label_list.value_counts().idxmax()

# 创建决策树，传入的是一个dataframe，最后一列为label
# PS：这里理解起来容易，自己写好难，看了下sklearn源码，迷迷糊糊的
def Create_Tree(data):
    feature = data.columns[:-1]
    label_list = data.iloc[:, -1]
    # 如果样本全属于同一类别C，将此节点标记为C类叶节点
    if len(pd.unique(label_list)) == 1:
        return label_list.values[0]
    # 如果待划分的属性集A为空，或者样本在属性A上取值相同，则把该节点作为叶节点，并标记为样本数最多的分类
    elif len(feature)==0 or len(data.loc[:,feature].drop_duplicates())==1:
        return get_most_label(label_list)
    # 从A中选择最优划分属性
    best_attr = get_max_gain(data)
    tree = {best_attr: {}}
    # 对于最优划分属性的每个属性值，生成一个分支
    for attr,gb_data in data.groupby(by=best_attr):
        if len(gb_data) == 0:
            tree[best_attr][attr] = get_most_label(label_list)
        else:
            # 在data中去掉已划分的属性
            new_data = gb_data.drop(best_attr,axis=1)
            # 递归构造决策树
            tree[best_attr][attr] = Create_Tree(new_data)
    return tree

# 获取树的叶子节点数目
def get_num_leafs(decision_tree):
    num_leafs = 0
    first_str = next(iter(decision_tree))
    second_dict = decision_tree[first_str]
    for k in second_dict.keys():
        if isinstance(second_dict[k], dict):
            num_leafs += get_num_leafs(second_dict[k])
        else:
            num_leafs += 1
    return num_leafs

'''
以下为画图函数
不是自己写的，网上找来改的
觉得太繁琐，而且实际用的过程中画图也是调用Graphiz包画树
所以在此就偷了个懒
改个代码了事，emmmmm
'''
# 获取树的深度
def get_tree_depth(decision_tree):
    max_depth = 0
    first_str = next(iter(decision_tree))
    second_dict = decision_tree[first_str]
    for k in second_dict.keys():
        if isinstance(second_dict[k], dict):
            this_depth = 1 + get_tree_depth(second_dict[k])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

# 绘制节点
def plot_node(node_txt, center_pt, parent_pt, node_type):
    arrow_args = dict(arrowstyle='<-')
    create_plot.ax1.annotate(node_txt, xy=parent_pt,  xycoords='axes fraction', xytext=center_pt, textcoords='axes fraction', va="center", ha="center", bbox=node_type,arrowprops=arrow_args)

# 标注划分属性
def plot_mid_text(cntr_pt, parent_pt, txt_str):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_str, va="center", ha="center", color='red')

# 绘制决策树
def plot_tree(decision_tree, parent_pt, node_txt):
    d_node = dict(boxstyle="sawtooth", fc="0.8")
    leaf_node = dict(boxstyle="round4", fc='0.8')
    num_leafs = get_num_leafs(decision_tree)
    first_str = next(iter(decision_tree))
    cntr_pt = (plot_tree.xoff + (1.0 +float(num_leafs))/2.0/plot_tree.totalW, plot_tree.yoff)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, d_node)
    second_dict = decision_tree[first_str]
    plot_tree.yoff = plot_tree.yoff - 1.0/plot_tree.totalD
    for k in second_dict.keys():
        if isinstance(second_dict[k], dict):
            plot_tree(second_dict[k], cntr_pt, k)
        else:
            plot_tree.xoff = plot_tree.xoff + 1.0/plot_tree.totalW
            plot_node(second_dict[k], (plot_tree.xoff, plot_tree.yoff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xoff, plot_tree.yoff), cntr_pt, k)
    plot_tree.yoff = plot_tree.yoff + 1.0/plot_tree.totalD

# 绘制图像
def create_plot(dtree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_num_leafs(dtree))
    plot_tree.totalD = float(get_tree_depth(dtree))
    plot_tree.xoff = -0.5/plot_tree.totalW
    plot_tree.yoff = 1.0
    plot_tree(dtree, (0.5, 1.0), '')
    plt.show()

if __name__ == "__main__":
    data = pd.read_excel('data.xlsx')   # 读取数据，存入dataframe中
    mytree = Create_Tree(data)  # 创建一个决策树模型
    create_plot(mytree)     # 画出决策过程