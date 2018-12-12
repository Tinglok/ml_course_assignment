import numpy as np
import random
import matplotlib.pyplot as plt


# 读取数据
def load_Data():
    # 设置随机数种子
    np.random.seed(2)
    # 随机生成服从正态分布的矩阵
    X = np.random.randn(300, 2)
    # 将生成的矩阵加上噪音
    for i in range(100):
        X[100:200, 0][i] = X[100:200, 0][i] + 3
        X[100:200, 1][i] = X[100:200, 1][i] + 6
        X[200:300, 0][i] = X[200:300, 0][i] + 8
        X[200:300, 1][i] = X[200:300, 1][i] + 4
    return X




# 随机采样函数
def ramdom_sample(dataSet, k):
    # 设置随机数种子
    random.seed(80)
    # 从数据集中随机选取k个数据返回
    dataSet = list(dataSet)
    return random.sample(dataSet, k)




# 计算欧氏距离
def compute_distance(vec1, vec2):
    # 计算向量1与向量2之间的欧式距离
    return np.sqrt(np.sum(np.square(vec1 - vec2)))



# 对每个属于dataSet的item， 计算item与centroidList中k个质心的距离，找出距离最小的，并将item加入相应的簇类中
def compute_min_distance(dataSet, centroidList):
    clusterDict = dict()  # dict保存簇类结果
    k = len(centroidList)
    for item in dataSet:
        vec1 = item
        flag = -1
        minDis = float("inf")  # 初始化最小距离为正无穷
        # 循环结束时， flag保存与当前item最近的蔟标记
        for i in range(k):
            vec2 = centroidList[i]
            distance = compute_distance(vec1, vec2)  # error
            if distance < minDis:
                minDis = distance
                flag = i
        if flag not in clusterDict.keys():
            clusterDict.setdefault(flag, [])
        clusterDict[flag].append(item)  # 将符合条件的点加到相应的簇类中
    return clusterDict



# 重新计算并更新k个质心
def compute_center(clusterDict):
    centroidList = []
    for key in clusterDict.keys():
        centroid = np.mean(clusterDict[key], axis=0)
        centroidList.append(centroid)
    return centroidList  # 得到新的质心



# 计算各蔟集合间的均方误差，将蔟类中各个向量与质心的距离累加求和
def compute_loss(centroidList, clusterDict):
    sum = 0.0
    for key in clusterDict.keys():
        vec1 = centroidList[key]
        distance = 0.0
        for item in clusterDict[key]:
            vec2 = item
            distance += compute_distance(vec1, vec2)
        sum += distance
    return sum



# 绘制初始数据图
def init_Graph(dataSet):
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c='r')
    plt.title('Kmeans-Times 0')
    plt.show()



# 展示聚类结果
def show_Cluster(centroidList, clusterDict, times):
    colorMark = ['or', 'ob', 'og']  # 不同簇类标记，o表示圆形
    centroidMark = ['r', 'g', 'b']  # 设置不同的类显示为不同的颜色

    # 绘制聚类结果图
    for key in clusterDict.keys():
        plt.plot(
            centroidList[key][0],
            centroidList[key][1],
            centroidMark[key],
            markersize=12)  # 绘制聚类质心点
        for item in clusterDict[key]:
            plt.plot(item[0], item[1], colorMark[key])  # 绘制样本点
    plt.title('Kmeans-Times ' + str(times))
    plt.show()


if __name__ == '__main__':
    dataSet = load_Data()  # 读取数据
    init_Graph(dataSet)  # 绘制初始数据散点图
    centroidList = ramdom_sample(dataSet, 3)  # 随机采三个样本数据，预设聚三类
    clusterDict = compute_min_distance(
        dataSet, centroidList)  # 计算距离最近的点，并存入字典中
    loss = compute_loss(centroidList, clusterDict)  # 计算两次聚类的误差
    old_loss = 1  # 将初始损失值设为1
    times = 0  # 记录聚类次数
    # 当两次聚类的误差小于阈值0.0001，说明质心基本确定
    while abs(loss - old_loss) >= 0.0001:
        centroidList = compute_center(clusterDict)
        clusterDict = compute_min_distance(dataSet, centroidList)
        old_loss = loss
        loss = compute_loss(centroidList, clusterDict)
        times += 1
        show_Cluster(centroidList, clusterDict, times)
