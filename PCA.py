import numpy as np
from sklearn.decomposition import PCA


'''
初始化一个矩阵
一共6个样本（x,y）
它们分布在直线y=x上的点，并且聚集在x=-3、-2、-1、1、2、3上
'''
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
_, feature = X.shape

# 设置将维度降到一维
dim = 1

# 按列求均值
X_mean=np.array([np.mean(X[:,i]) for i in range(feature)])
# 将所有样本进行中心化
X_norm=X-X_mean

# 求样本的协方差矩阵XX_T
conv_matrix=np.dot(np.transpose(X_norm),X_norm)

# 计算特征值和特征向量
eig_val, eig_vec = np.linalg.eig(conv_matrix)
# 将特征值及其对应的特征向量存在列表中
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(feature)]
# 将列表按特征值从大到小排序
eig_pairs.sort(reverse=True)
# 选取第一个特征值对应的特征向量
feature = np.array([eig_pairs[dim-1][1]])

# 计算出降维后的数据
data = np.dot(X_norm,np.transpose(feature))
print("\n降维后的数据为：\n", data)
print("---------------------------------")


'''
与sklearn包中的PCA方法对比，将数据降成一维
观察可发现与手写的降维结果相同
'''
pca = PCA(n_components=1)
newData = pca.fit_transform(X)
print("调用sklearn后降维的数据为：\n", newData)