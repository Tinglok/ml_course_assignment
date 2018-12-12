import math
import matplotlib.pyplot as plt
import numpy as np


# 计算欧几里得距离,a,b分别为两个元组
def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# 计算距离矩阵中的最小值
def dist_min(Ci, Cj):
    return min(dist(i, j) for i in Ci for j in Cj)

# 找出距离矩阵的最小值下标
def find_Min(M):
    min = 1000
    x = 0
    y = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i != j and M[i][j] < min:
                min = M[i][j]
                x = i
                y = j
    return x, y, min

# 层次聚类模型
def AGNES(dataset, k):
    # 初始化C和M
    C = []
    M = []
    for i in dataset:
        Ci = []
        Ci.append(i)
        C.append(Ci)
    for i in C:
        Mi = []
        for j in C:
            Mi.append(dist_min(i, j))
        M.append(Mi)
    q = len(dataset)
    # 合并更新
    while q > k:
        x, y, min = find_Min(M)
        C[x].extend(C[y])  # 合并C[x]、C[y]
        del (C[y])   # 删除C[y]
        M = []
        for i in C:
            Mi = []
            for j in C:
                Mi.append(dist_min(i, j))
            M.append(Mi)
        q -= 1
    return C

# 绘制层次聚类后的图
def show_Graph(C):
    colValue = ['r', 'g', 'b']
    for i in range(len(C)):
        cor_X = []  # x坐标列表
        cor_Y = []  # y坐标列表
        for j in range(len(C[i])):
            cor_X.append(C[i][j][0])
            cor_Y.append(C[i][j][1])
        plt.scatter(cor_X, cor_Y, marker='x', color=colValue[i % len(colValue)], label=i)
        plt.title('AFTER-AGNES')
    plt.legend(loc='upper right')
    plt.show()

# 初始化数据
def load_Data():
    # 设置随机数种子
    np.random.seed(2)
    # 随机生成服从正态分布的矩阵
    X = np.random.randn(30, 2)
    # 将生成的矩阵加上噪音
    for i in range(10):
        X[10:20, 0][i] = X[10:20, 0][i] + 3
        X[10:20, 1][i] = X[10:20, 1][i] + 6
        X[20:30, 0][i] = X[20:30, 0][i] + 8
        X[20:30, 1][i] = X[20:30, 1][i] + 4
    return X

# 绘制初始数据图
def init_Graph(dataSet):
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c='r')
    plt.title('BEFORE-AGNES')
    plt.show()

if __name__ == '__main__':
    dataSet = load_Data()  # 读取数据
    init_Graph(dataSet)  # 绘制初始数据散点图
    C = AGNES(dataSet, 3)
    show_Graph(C)