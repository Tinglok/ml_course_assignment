import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogisticRegression:
    # 处理得到的数据，并划分为train_set和test_set
    def feature_engineering(self, datas, train_prob):
        # 将花的类型转换为0，1，2
        class_mapping = {
            label: idx for idx,
            label in enumerate(
                np.unique(
                    datas['Name']))}
        datas['Name'] = datas['Name'].map(class_mapping)

        # 要求二分类，故只留下类型0，1
        class_mapping_1 = {0: 0, 1: 1}
        datas['Name'] = datas['Name'].map(class_mapping_1)
        datas = datas.dropna()

        # 选取两列做feature，最后一列做label
        X = datas.iloc[:, 0:2]
        y = datas.iloc[:, -1]
        X = X.values

        # 标准化数据
        X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
        X = X.tolist()

        # 打乱数据
        X, y = shuffle(X, y)
        sample = len(X)

        # 将数据分为训练集和测试集，并转换成矩阵
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, train_size=train_prob)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        x_1 = x_train[:, 0]
        x_2 = x_train[:, 1]

        x_1 = np.array(x_1)
        x_2 = np.array(x_2)

        x_1 = x_1.reshape(int(train_prob * sample), 1)
        x_2 = x_2.reshape(int(train_prob * sample), 1)

        y_train = y_train.reshape(int(train_prob * sample), 1)

        # 将样本数和训练样本占比存入self中以备后用
        self.sample = sample
        self.train_prob = train_prob

        return x_1, x_2, y_train, x_test, y_test

    # 激活函数sigmoid
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    # 梯度下降法计算最优参数
    def fit(self, x_1, x_2, y_train, iter_num=10000, alpha=0.0001):
        # 计算训练样本的个数
        m = int(self.sample * self.train_prob)

        # 初始化三个theta
        theta_0 = np.zeros((m, 1))
        theta_1 = np.zeros((m, 1))
        theta_2 = np.zeros((m, 1))

        # 使用批量梯度下降法
        epochs = 0
        cost_func = []
        while (epochs < iter_num):
            # 计算此时的y_hat
            y = theta_0 + theta_1 * x_1 + theta_2 * x_2
            y = self.sigmoid(y)

            '''
            均方误差（MSE）是非凸的，直接拿来当代价函数进行梯度下降训练得到的点有可能是局部收敛点
            因此时为二分类，故使用以下方法作为代价函数
            此时最终训练得到的点才是全局收敛点
            '''
            cost = (- np.dot(np.transpose(y_train), np.log(y)) -
                    np.dot(np.transpose(1 - y_train), np.log(1 - y))) / m

            # 此梯度由上述代价函数求偏导得来，推导过程较复杂故在此省略
            theta_0_grad = np.dot(np.ones((1, m)), y - y_train) / m
            theta_1_grad = np.dot(np.transpose(x_1), y - y_train) / m
            theta_2_grad = np.dot(np.transpose(x_2), y - y_train) / m

            # 求出训练后的参数，alpha为学习率
            theta_0 = theta_0 - alpha * theta_0_grad
            theta_1 = theta_1 - alpha * theta_1_grad
            theta_2 = theta_2 - alpha * theta_2_grad

            cost_func.append(cost)
            epochs += 1

            # 将各参数存入self中以备后用
            self.cost_func = cost_func
            self.theta_0 = theta_0
            self.theta_1 = theta_1
            self.theta_2 = theta_2
            self.iter_num = iter_num

    # 预测函数，得到预测准确率
    def predict(self, x_test, y_test):
        # 处理test_set，并转换为矩阵
        test_x_1 = x_test[:, 0]
        test_x_2 = x_test[:, 1]

        test_x_1 = np.array(test_x_1)
        test_x_2 = np.array(test_x_2)

        # 此步骤用了np.float32是因为使用默认值会出现精度缺失问题，导致结果出错
        test_x_1 = test_x_1.reshape(
            int(self.sample * np.float32(1 - self.train_prob)), 1)
        test_x_2 = test_x_2.reshape(
            int(self.sample * np.float32(1 - self.train_prob)), 1)

        # 由于参数是根据训练集训练得到的，故shape不匹配测试集，故需要将参数的shape调整至符合测试集
        index = list(range(int(self.sample * np.float32(1 -
                                                        self.train_prob)), int(self.sample * self.train_prob)))

        theta_0 = np.delete(self.theta_0, index)
        theta_1 = np.delete(self.theta_1, index)
        theta_2 = np.delete(self.theta_2, index)

        theta_0 = theta_0.reshape(
            int(self.sample * np.float32(1 - self.train_prob)), 1)
        theta_1 = theta_1.reshape(
            int(self.sample * np.float32(1 - self.train_prob)), 1)
        theta_2 = theta_2.reshape(
            int(self.sample * np.float32(1 - self.train_prob)), 1)

        # 将训练得到的参数代入，计算得到预测值
        y_pred = theta_0 + theta_1 * test_x_1 + theta_2 * test_x_2
        y_pred = self.sigmoid(y_pred)

        # 为画分类图考虑，将三个参数合成一个矩阵
        weights = np.array([theta_0[0][0], theta_1[0][0], theta_2[0][0]])
        self.weights = weights

        # 将阈值设为0.5，大于或等于0.5则预测为1，小于0.5则为0
        y_pred_1 = []
        for val in y_pred:
            if (val >= 0.5):
                y_pred_1.append(1)
            else:
                y_pred_1.append(0)
        print(
            'The accuracy of the score is: ',
            accuracy_score(
                y_test,
                y_pred_1))

    # 随着epoch迭代，Training loss变化图
    def plot_training_loss(self):
        cost_func = np.array(self.cost_func)
        cost_func = cost_func.reshape(self.iter_num, 1)
        plt.xlabel('epochs')
        plt.ylabel('Training-loss')
        plt.plot(range(len(cost_func)), cost_func)

    # 分析数据，画出决策边界
    def plot_decision_region(self, feature, label):
        # 将矩阵转化为数组
        dataArr = np.array(feature)

        n = feature.shape[0]
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []

        for i in range(n):
            if int(label[i]) == 1:
                xcord1.append(dataArr[i, 0])
                ycord1.append(dataArr[i, 1])
            else:
                xcord2.append(dataArr[i, 0])
                ycord2.append(dataArr[i, 1])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c="green")

        # x为numpy.arange格式，并且以0.1为步长从-3.0到3.0
        x = np.arange(-3.0, 3.0, 0.1)

        # 拟合曲线为0 = w0 * x0 + w1 * x1 + w2 * x2, 故x2 = (-w0 * x0 - w1 * x1) /
        # w2, x0为1, x1为x, x2为y, 故有
        y = (-self.weights[0] - self.weights[1] * x) / self.weights[2]

        ax.plot(x, y)
        ax.legend(
            labels=[
                'fitting-curve',
                'Iris-setosa',
                'Iris-versicolor'],
            loc='best')
        plt.xlabel("Sepal length")  # X轴的标签
        plt.ylabel("Sepal width")  # Y轴的标签
        plt.show()


if __name__ == "__main__":
    iris = pd.read_csv('iris.csv')
    lr = LogisticRegression()  # 创建实例对象lr
    x_1, x_2, y_train, x_test, y_test = lr.feature_engineering(
        iris, train_prob=0.9)  # 数据预处理
    lr.fit(x_1, x_2, y_train, iter_num=10000, alpha=0.01)  # 训练模型
    lr.predict(x_test, y_test)  # 得到预测结果
    lr.plot_training_loss()  # 画出loss-epoch图
    lr.plot_decision_region(feature=x_test, label=y_test)  # 画出决策边界
