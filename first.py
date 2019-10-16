import numpy as np
import csv
import os
from sklearn.model_selection import train_test_split
#近邻函数类
from sklearn.neighbors import KNeighborsClassifier
#交叉检测
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt


#导入数据集

data_folder = os.path.join("D:\python代码\Estimator")
data_filename = os.path.join(data_folder, "ionosphere.data")
print(data_filename)


#创建一个351行34列的矩阵来存数据（数据集大小知道）
#x为数据集，y为分类集
X = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

#存储数据集
with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        # Get the data, converting each item to a float
        data = [float(datum) for datum in row[:-1]]
        # Set the appropriate row in our dataset
        X[i] = data
        # 如果类别为“g”，值为1， 否则值为0。
        y[i] = row[-1] == 'g'

#将样本划分为测试集和训练集
# 训练集Xd_train和测试集Xd_test
#  y_train和y_test 分别为以上两个数据集的类别信息
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
print("训练子集样本容量为 {}".format(X_train.shape[0]))
print("测试子集样本容量为 {}".format(X_test.shape[0]))
print("有{}个种类".format(X_train.shape[1]))

#近邻算法初始化estimator实例，默认选择5个近邻作为分类依据
estimator = KNeighborsClassifier()

#利用fit函数传入训练集对其训练
estimator.fit(X_train, y_train)

#用传入测试集给predict函数测试效果
y_predicted = estimator.predict(X_test)

#对正确率评估，
# np.mean计算每一列的均值，比较y_test和y_predicted一样取1，否则返回0，再取均值
accuracy = np.mean(y_test == y_predicted) * 100

print("准确率为 {0:.1f}%".format(accuracy))

#交叉检测
scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print("交叉检测后训练后准确率为 {0:.1f}%".format(average_accuracy))


#测试KNeighborsClassifier参数为几准确率最高
avg_scores = []
all_scores = []
#在1到20里面看哪个合适
parameter_values = list(range(1, 21))
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

#建一个窗口
plt.figure(figsize=(20,10))
#x轴数据，y轴数据，-o：颜色样式，
# LineWidth——指定线宽，
# MarkerEdgeColor——指定标识符的边缘颜色
#MarkerFaceColor——指定标识符填充颜色
#MarkerSize——指定标识符的大小
plt.plot(parameter_values, avg_scores, '-o', linewidth=5, markersize=24)
plt.show()


