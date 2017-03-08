# -*- coding:utf-8 -*-
# author: yushan
# date: 2017-03-08


from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']
labels = data['target_names'][data['target']]
# features:
# [[ 4.9  3.   1.4  0.2]
#  [ 4.7  3.2  1.3  0.2]
#  [ 4.6  3.1  1.5  0.2]
#  [ 5.   3.6  1.4  0.2]
#  [ 5.4  3.9  1.7  0.4]
#  [ 4.6  3.4  1.4  0.3]
#  [ 5.   3.4  1.5  0.2]
#  [ 4.4  2.9  1.4  0.2]
#  [ 4.9  3.1  1.5  0.1]]
# feature_names:
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# 绘图
# for t,marker,c in zip(xrange(3),">ox","rgb"):
#     plt.scatter(features[target == t,0],
#                 features[target == t,1],
#                 marker=marker,
#                 c=c)
# plt.autoscale(tight=True)
# plt.show()
plength = features[:,2]
#用numpy操作来获取setosa的特征
is_setosa = (target == 0)
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()

# 打印结果：
# Maximum of setosa:1.9.
# minimum of others:3.0.
print ('Maximum of setosa:{0}.'.format(max_setosa))
print ('minimum of others:{0}.'.format(min_non_setosa))

# 简单的模型：花瓣长度小于2，为Setosa,否则，为其他
# 某个维度上的简单阈值
# if features[:,2] < 2:print 'Iris Setosa'
# else:print 'Iris Virginica or Iris Versicolour'

features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica')

# 对所有可能的特征和阈值进行遍历，找出更高正确率。
# 正确率指模型正确分类的那部分样本所占的比例
# 运行结果：
# best_acc = 0.94
# best_fi = 3
# best_t = 1.6
best_acc = -1.0
for fi in xrange(features.shape[1]):
    thresh = features[:,fi].copy()
    thresh.sort()
    for t in thresh:
        pred = (features[:,fi] > t)
        acc = (pred == virginica).mean()
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t

# if example[best_fi] > t: print 'virginica'
# else:print 'versicolor'

# 最邻近分类
def distance(p0,p1):
    'Computes Squsred euclidean dictance'
    return np.sum((p0 - p1)**2)

def nn_classify(training_set,training_labels,new_example):
    dists = np.array([distance(t,new_example) for t in training_set])
    nearest = dists.argmin()
    return training_labels[nearest]