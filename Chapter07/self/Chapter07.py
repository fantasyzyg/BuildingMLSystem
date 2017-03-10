# -*- coding:utf-8 -*-
# author: yushan
# date: 2017-03-10


import numpy as np
from sklearn.datasets import load_boston
import pylab as plt

# 一维回归
boston = load_boston()
plt.scatter(boston.data[:,5], boston.target,color='r')
plt.xlabel("RM")
plt.ylabel("House Price")


x = boston.data[:,5]
xmin = x.min()
xmax = x.max()
x = np.array([[v,1] for v in x])
y = boston.target

(slope,bias),res,_,_ = np.linalg.lstsq(x,y)
plt.plot([xmin,xmax],[slope*xmin + bias, slope*xmax + bias], '-', lw=4)


rmse = np.sqrt(res[0]/len(x))
print('Residual: {}'.format(rmse))
plt.show()

# 多维回归
boston = load_boston()
x = np.array([np.concatenate((v,[1])) for v in boston.data])
y = boston.target
s,total_error,_,_ = np.linalg.lstsq(x,y)

rmse = np.sqrt(total_error[0]/len(x))
print('Residual: {}'.format(rmse))

plt.plot(np.dot(x,s), boston.target,'ro')
plt.plot([0,50],[0,50], 'g-')
plt.xlabel('predicted')
plt.ylabel('real')
plt.show()

# 交叉验证
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
lr = LinearRegression(fit_intercept=True)
lr.fit(x,y)
p = np.array([lr.predict(xi) for xi in x])
e = p - y
total_error = np.sum(e*e)
rmse_train = np.sqrt(total_error/len(p))
print "rmse_train=",rmse_train

kf = KFold(len(x), n_folds=10)
err = 0
for train,test in kf:
    lr.fit(x[train],y[train])
    p = np.array([lr.predict(xi) for xi in x[test]])
    e = p-y[test]
    err += np.sum(e*e)

rmse_10cv = np.sqrt(err/len(x))
print('RMSE on 10-fold CV: {}'.format(rmse_10cv))

# 使用LassoCV，RidgeCV，ElasticNetCV
from sklearn.linear_model import ElasticNetCV
met = ElasticNetCV(fit_intercept=True)
kf = KFold(len(target),n_folds=10)
for train,test in kf:
    met.fit(data[train],target[train])
    p = map(met.predict,data[test])
    p = np.array(p).ravel()
    e = p - target[test]
    err += np.dot(e,e)

rmse_10cv = np.sqrt(err/len(target))