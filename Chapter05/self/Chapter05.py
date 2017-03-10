# -*- coding:utf-8 -*-
# author: yushan
# date: 2017-03-06


import re
import numpy as np
from sklearn import neighbors
from sklearn.cross_validation import KFold
knn = neighbors.KNeighborsClassifier(n_neighbors=2)
# print knn
# x = [[1],[2],[3],[4],[5],[6]]
# y = [0,0,0,1,1,1]
# knn.fit(x,y)
# print knn.predict(1.5)
# print knn.predict(37)
# print knn.predict(3)
# print knn.predict_proba(1.5)
# print knn.predict_proba(37)
# print knn.predict_proba(3.5)

code_match = re.compile('<pre>(.*?)</pre>',re.MULTILINE|re.DOTALL)
link_match = re.compile('<a href="http://.*?".*?>(.*?)</a>',re.MULTILINE|re.DOTALL)

def extract_features_from_body(s):

    link_count_in_code = 0

    # 统计代码中的链接，并提取
    for match_str in code_match.findall(s):
        link_count_in_code += len(link_match.findall(match_str))

    return len(link_match.findall(s)) - link_count_in_code

X = np.asarray([extract_features_from_body(text) for post_id,text in fetch_posts if post_id in all_answers])
knn = neighbors.KNeighborsClassifier()
knn.fit(X,Y)

# 评价分类器性能
scores = []
cv = KFold(n=len(X),n_folds=10,indices=True)
for train,test in cv:
    X_train,y_train = X[train],Y[train]
    X_test,y_test = X[test],Y[test]
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X,Y)
    scores.append(clf.score(X_test,y_test))

print ("Mean(scores)=%.5f\tStddev(scores)=%.5f"%(np.mean(scores),np.std(scores)))

# 提取更多特征
def extract_features_from_body(s):

    num_code_lines = 0
    link_count_in_code = 0
    code_free_s = 0

    for match_str in code_match.findall(s):
        num_code_lines += match_str.count('\n')
        code_free_s += code_match.sub("",code_free_s)


    link_count_in_code += len(link_match.findall(match_str))
    links = link_match.findall(s)
    link_count = len(links)
    link_count -= link_count_in_code
    html_free_s = re.sub(" +"," ",tag_match.sub('',code_free_s)).replace("\n","")
    link_free_s = html_free_s
    for anchor in anchors:
        if anchor.lower().startswith("http://"):
            link_free_s = link_free_s.replace(anchor,"")

        num_text_tokens = html_free_s.count(" ")

    return num_text_tokens,num_code_lines,link_count
# 提取之后，按照上面的K近邻方法重新训练模型

# 拓展特征空间：AvgSentLen,AvgWordLen,NumAllCaps,NumExclams;重新训练模型

"""
提升模型效果的方法：
1、增加更多的数据；
2、考虑模型的复杂度，改变K值；
3、修改特征空间，设置新的特征等等；
4、改变模型；
"""
"""
偏差-方差及其折中：
解决高偏差：增加更多的特征，尝试别的模型
解决高方差：尝试更多的数据，降低模型复杂度，删减一些特征
高偏差或低偏差：增大K或者削减特征空间
"""

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
# print clf
clf.fit(X,y)
print (np.exp(clf.intercept_),np.exp(clf.coef_.ravel()))
def lr_model(clf,X):
    return 1/(1+np.exp(-(clf.intercept_+clf.coef_*X)))
print ("P(x=-1)=%.2f\tP(x=7)=%.2f"%lr_model(clf,-1),lr_model(clf,7))

"""
正确率与召回率
"""
from sklearn.metrics import precision_recall_curve
precision,recall,thresholds = precision_recall_curve(y_test,clf.predict(X_test))

thresholds = np.hstack(([0],thresholds[medium]))
idx80 = precision >= 0.8
print ("P=%.2f R=%.2f thresh=%.2f"%(precision[idx80][0],recall[idx80][0],thresholds[idx80][0]))

from sklearn.metrics import classification_report
print classification_report(y_test,clf.predict_proba[:,1]>0.63,target_names=['not accepted','accepted'])

import pickle
pickle.dump(clf,open("logreg.dat","w"))
clf = pickle.load(open("logreg.dat","r"))