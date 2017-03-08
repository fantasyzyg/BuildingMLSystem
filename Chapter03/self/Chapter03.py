# -*- coding:utf-8 -*-
# author: yushan
# date: 2017-03-08


import os
import sys
import scipy as sp
import math

# 统计词语
DIR = 'toy/'
posts = [open(os.path.join(DIR,f)).read() for f in os.listdir(DIR)]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
# content = ["How to format my hard disk","Hard disk format problems"]
# X = vectorizer.fit_transform(content)
# print vectorizer.get_feature_names()
# print (X.toarray().transpose())
X_train = vectorizer.fit_transform(posts)
num_samples,num_features = X_train.shape
print num_samples
print num_features
print vectorizer.get_feature_names()
new_post = "imaging databases"
new_post_vc = vectorizer.transform([new_post])
print new_post_vc.toarray()
def dist_raw(v1,v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

best_doc = None
best_dist = sys.maxint
best_i = None
for i in range(0,num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_raw(post_vec,new_post_vc)
    print "=== Post %i with dist=%.2f: %s"%(i,d,post)
    if d<best_dist:
        best_dist = d
        best_i = i

print ("Best post is %i with dist=%.2f"%(best_i,best_dist))
# print X_train.getrow(3).toarray()
# print X_train.getrow(4).toarray()

# 词语频次向量归一化
def dist_norm(v1,v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

for i in range(0,num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_norm(post_vec,new_post_vc)
    print "=== Post %i with dist=%.2f: %s"%(i,d,post)
    if d<best_dist:
        best_dist = d
        best_i = i

print ("Best post is %i with dist=%.2f"%(best_i,best_dist))

# 删除不重要的词语，停用词等
vectorizer = CountVectorizer(min_df=1,stop_words='english')
X_train = vectorizer.fit_transform(posts)
num_samples,num_features = X_train.shape
print num_samples
print num_features
print vectorizer.get_feature_names()
new_post_vc = vectorizer.transform([new_post])
for i in range(0,num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_norm(post_vec,new_post_vc)
    print "=== Post %i with dist=%.2f: %s"%(i,d,post)
    if d<best_dist:
        best_dist = d
        best_i = i

print ("Best post is %i with dist=%.2f"%(best_i,best_dist))

# 词干处理
import nltk.stem
s = nltk.stem.SnowballStemmer('english')
print s.stem("graphics")
print s.stem("image")
print s.stem("imaging")

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer,self).build_analyzer()
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedCountVectorizer(min_df=1,stop_words='english')
X_train = vectorizer.fit_transform(posts)
print vectorizer.get_feature_names()
new_post_vc = vectorizer.transform([new_post])
for i in range(0,num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_norm(post_vec,new_post_vc)
    print "=== Post %i with dist=%.2f: %s"%(i,d,post)
    if d<best_dist:
        best_dist = d
        best_i = i

print ("Best post is %i with dist=%.2f"%(best_i,best_dist))

# 停用词兴奋剂
def tfidf(term,doc,docset):
    tf = float(doc.count(term)) / sum(doc.count(term) for doc in docset)
    idf = math.log(float(len(docset))/(len([doc for doc in docset])))
    return tf*idf

from sklearn.feature_extraction.text import TfidfVectorizer
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer,self).build_analyzer()
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=1,stop_words='english',charset_error='ignore')