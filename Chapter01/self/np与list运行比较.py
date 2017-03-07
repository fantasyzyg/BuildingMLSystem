# -*- coding:utf-8 -*-
# author: yushan
# date: 2017-03-07


# 比较Numpy与Python列表的运行速度，相同运算重复10000次
"""
第一次运行结果：
Normal Python:0.914836 sec
Naive Numpy:2.000678 sec
Good Numpy:0.021106 sec
第二次运行结果：
Normal Python:0.922085 sec
Naive Numpy:2.036564 sec
Good Numpy:0.022636 sec
其中，naive_np_sec与书上运行结果相差2秒左右，
"""
import timeit
normal_py_sec = timeit.timeit('sum(x*x for x in xrange(1000))',
                              number=10000)
naive_np_sec = timeit.timeit('sum(na*na)',
                             setup="import numpy as np; na=np.arange(1000)",
                             number=10000)
good_np_sec = timeit.timeit('na.dot(na)',
                            setup='import numpy as np; na=np.arange(1000)',
                            number=10000)

print ("Normal Python:%f sec" % normal_py_sec)
print ("Naive Numpy:%f sec" % naive_np_sec)
print ("Good Numpy:%f sec" % good_np_sec)