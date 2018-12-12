import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
# 这个包如果没有就直接pip install mlxtend即可
from mlxtend.plotting import plot_decision_regions

# 读取数据
iris = pd.read_csv('iris.csv')

# 将花的类型转换为0，1，2
class_mapping = {
    label: idx for idx,
    label in enumerate(
        np.unique(
            iris['Name']))}
iris['Name'] = iris['Name'].map(class_mapping)

# 将数据集划分为feature和label
X = iris.iloc[:, 0:4]
y = iris.iloc[:, -1]

# 将数据的feature降成两维
pca = PCA(n_components=2)
X = pca.fit_transform(X)

# 将label转换成np.array形式，方便画图
y = np.array(y)

# 将数据划分为训练集和测试集，训练集占比70%
x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=0)

# 利用管道，将数据集先进行标准化处理，后进行svm高斯核化分类
Bayes_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("Gaussian", GaussianNB())
])

# 将训练集导入模型进行训练
Gaussian = Bayes_clf.fit(x_train, y_train)

# 将测试集的feature输入到模型中进行预测
y_pred = Gaussian.predict(x_test)

# 输出预测精度
print("The accuracy of the svm model is:", accuracy_score(y_test, y_pred))

# 画出决策边界
plot_decision_regions(X, y, clf=Gaussian, legend=2,
                      X_highlight=x_test)

# 绘制标题，横纵坐标
plt.xlabel('first principal components')
plt.ylabel('second principal components')
plt.title('Bayes on Iris')
plt.show()
