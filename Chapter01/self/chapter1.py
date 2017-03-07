# -*- coding:utf-8 -*-
# author: yushan
# date: 2017-03-07


# 对Web访问请求进行拟合
import os
import scipy as sp
import matplotlib.pyplot as plt

# 读取数据
data = sp.genfromtxt('web_traffic.tsv',deletechars="\t")
# print data[:10]
# print data.shape

# 对数据进行预处理和清洗
x = data[:,0] # 取第一列数据
y = data[:,1] # 取第二列数据
print "无效数据的个数：",sp.sum(sp.isnan(y))
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

# 描绘数据,书上的程序与配套的例程有些不一样的地方，会同时做出注释

colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']

# 书上的程序，配套例程将绘图程序写成函数，直接调用，关键代码是类似的
# plt.scatter(x,y)
# plt.title("Web traffic over the last month")
# plt.xlabel("Time")
# plt.ylabel("Hits/hour")
# plt.xticks([w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])
# plt.autoscale(tight=True)
# plt.grid()
# plt.show()

# 配套例程中的绘图函数
def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks(
        [w * 7 * 24 for w in range(10)],
        ['week %i' % w for w in range(10)])

    if models:
        if mx is None:
            mx = sp.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, linestyles, colors):
            # print "Model:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig(fname) # 将图像保存到本地

# 原始数据绘图
# plot_models(x, y, None, os.path.join("..", "01_01.png"))

# 误差计算
def error(f,x,y):
    return sp.sum((f(x) - y)**2)

# 多项式拟合。一阶,最小二乘法
"""
拟合结果：
fp1:       [   2.59619213  989.02487106]
residuals: [  3.17389767e+08]
"""
fp1, residuals, rank, sv, rcond = sp.polyfit(x,y,1,full=True)
print ("Model parameters: %s" % fp1)
print residuals

f1 = sp.poly1d(fp1)
print error(f1,x,y) # 结果：317389767.34

# 绘制一阶拟合直线
fx = sp.linspace(0,x[-1],1000)
plt.plot(fx,f1(fx),linewidth=4)
plt.legend(["d=%i"%f1.order],loc="upper left")

f2p = sp.polyfit(x,y,2)
print f2p           # 二阶结果：[  1.05322215e-02  -5.26545650e+00   1.97476082e+03]
f2 = sp.poly1d(f2p)
print error(f2,x,y) # 二阶误差：179983507.878
f3p = sp.polyfit(x,y,3)
"""
三阶结果： [  3.04960132e-05  -2.35893797e-02   4.94226019e+00   1.33328797e+03]
误差： 139350144.032
十阶结果： [ -3.73981969e-22   1.36473757e-18  -2.14294407e-15   1.89884971e-12
  -1.04570108e-09   3.70867732e-07  -8.45604590e-05   1.19167041e-02
  -9.41618608e-01   3.33703840e+01   1.26421204e+03]
误差： 121942326.363
百阶结果：
[  0.00000000e+000   0.00000000e+000  -0.00000000e+000   0.00000000e+000
  -0.00000000e+000  -0.00000000e+000   0.00000000e+000  -0.00000000e+000
  -0.00000000e+000  -0.00000000e+000  -0.00000000e+000   0.00000000e+000
   0.00000000e+000  -0.00000000e+000  -0.00000000e+000  -0.00000000e+000
  -0.00000000e+000  -0.00000000e+000  -0.00000000e+000  -0.00000000e+000
   0.00000000e+000   0.00000000e+000  -0.00000000e+000   0.00000000e+000
   0.00000000e+000   0.00000000e+000   0.00000000e+000   0.00000000e+000
   0.00000000e+000  -0.00000000e+000   0.00000000e+000  -0.00000000e+000
  -0.00000000e+000   0.00000000e+000   0.00000000e+000   0.00000000e+000
   0.00000000e+000   0.00000000e+000   0.00000000e+000   0.00000000e+000
   0.00000000e+000   0.00000000e+000   0.00000000e+000   0.00000000e+000
   0.00000000e+000   0.00000000e+000   0.00000000e+000  -6.72743552e-140
   1.19703609e-136  -4.65674251e-135  -4.66492996e-131  -2.54251767e-128
   2.25949968e-126   1.39012293e-122   1.16111847e-119   4.32295749e-117
  -1.50787093e-114  -3.76689500e-111  -3.20563666e-108  -1.48599758e-105
   6.29861284e-104   8.53710630e-100   9.11080088e-097   5.56876672e-094
   1.31776280e-091  -1.52904243e-088  -2.41429118e-085  -1.84986745e-082
  -7.30002741e-080   2.08140641e-077   6.26297636e-074   5.61000411e-071
   2.52773871e-068  -4.42198925e-066  -1.85474943e-062  -1.62946634e-059
  -5.76714065e-057   3.55008747e-054   6.50708003e-051   3.79810405e-048
  -4.37206618e-046  -2.48052795e-042  -1.57870887e-039   3.45922468e-037
   1.09215889e-033   3.34469126e-031  -4.93774937e-028  -2.97864531e-025
   2.64859770e-022   1.18818541e-019  -2.02244897e-016   1.03623619e-013
  -2.90544445e-011   4.79746361e-009  -4.21055131e-007   5.98195142e-006
   2.59210916e-003  -2.60189823e-001   1.03506849e+001  -1.60103561e+002
   2.14972433e+003]
误差： 109452404.409
"""
# print "三阶结果：",f3p
# f3 = sp.poly1d(f3p)
# print "误差：",error(f3,x,y)
# f10p = sp.polyfit(x,y,10)
# print "十阶结果：",f10p
# f10 = sp.poly1d(f10p)
# print "误差：",error(f10,x,y)
# f100p = sp.polyfit(x,y,100)
# print "百阶结果：",f100p
# f100 = sp.poly1d(f100p)
# print "误差：",error(f100,x,y)

# plt.autoscale(tight=True)
# plt.grid()
# plt.show()

# 分段分析数据
inflection = 3.5 * 7 * 24
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa,ya,1))
fb = sp.poly1d(sp.polyfit(xb,yb,1))

fa_error = error(fa,xa,ya)
fb_error = error(fb,xb,yb)

print ("Error inflection= %f" % (fa_error+fb_error)) # 结果：Error inflection= 132950348.197616

fa2 = sp.poly1d(sp.polyfit(xa,ya,2))
fb2 = sp.poly1d(sp.polyfit(xb,yb,2))

fa2_error = error(fa2,xa,ya)
fb2_error = error(fb2,xb,yb)

print fb2_error
print fa2_error
print ("Error inflection= %f" % (fa_error+fb_error))

# 抽取部分数据进行拟合
frac = 0.3
split_idx = int(frac * len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])
fbt1 = sp.poly1d(sp.polyfit(xb[train], yb[train], 1))
fbt2 = sp.poly1d(sp.polyfit(xb[train], yb[train], 2))
fbt3 = sp.poly1d(sp.polyfit(xb[train], yb[train], 3))
fbt10 = sp.poly1d(sp.polyfit(xb[train], yb[train], 10))
fbt100 = sp.poly1d(sp.polyfit(xb[train], yb[train], 100))

"""
Error d=1: 6248249.315553
Error d=2: 5697536.167178
Error d=3: 5697576.239372
Error d=10: 6248938.994819
Error d=53: 7196334.219257
"""
for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
    print("Error d=%i: %f" % (f.order, error(f, xb[test], yb[test])))

# 利用误差最小的拟合函数进行预测
"""
fbt2:
         2
0.08615 x - 93.61 x + 2.711e+04
fbt2-100000:
         2
0.08615 x - 93.61 x - 7.289e+04

100,000 hits/hour expected at week 9.592736
"""
from scipy.optimize import fsolve
print fbt2
print fbt2-100000
reached_max = fsolve(fbt2-100000,800)/(7*24)
print("100,000 hits/hour expected at week %f" % reached_max[0])