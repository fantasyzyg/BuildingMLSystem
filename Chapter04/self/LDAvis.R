library("rJava")
library(RTextTools)
library(e1071)
library("Rwordseg")
library(jiebaR)
library(wordcloud2)
installDict(dictpath='D:/FTP/RData/computer.scel',dictname="computer",dicttype="scel")

# 使用时需要修改路径
train <- read.csv("D:/FTP/RData/commenttrain.csv", sep = ",", header = T, stringsAsFactors = F) 
#一级清洗——去标点  
sentence <- as.vector(train$msg)   
sentence <- gsub("[[:digit:]]*", "", sentence) #清除数字[a-zA-Z]  
sentence <- gsub("[a-zA-Z]", "", sentence)  
sentence <- gsub("\\.\\.", "", sentence)  
#二级清洗——去内容  
train <- train[!is.na(sentence), ]  
sentence <- sentence[!is.na(sentence)]  
train <- train[!nchar(sentence) < 2, ]  
sentence <- sentence[!nchar(sentence) < 2]
system.time(x <- segmentCN(strwords = sentence))   
temp <- lapply(x, length)  
temp <- unlist(temp)  
id <- rep(train[, "id"], temp)  
label <- rep(train[, "label"], temp)  
term <- unlist(x)  
trainterm <- as.data.frame(cbind(id, term, label), stringsAsFactors = F) 
stopwords <- read.csv("D:/FTP/RData/sentiment/stopword.csv")
stopwords <- stopwords[!stopwords$term %in% dict,]  
trainterm <- trainterm[!trainterm$term %in% stopwords,]
comments <- as.list(trainterm$term)
doc.list <- strsplit(as.character(comments),split=" ")
term.table <- table(unlist(doc.list)) 
term.table <- sort(term.table, decreasing = TRUE)
del <- term.table < 5| nchar(names(term.table))<2   #把不符合要求的筛出来
term.table <- term.table[!del]   #去掉不符合要求的
wordFre <- data.frame(term.table)
wordcloud2(wordFre[1:100,1:2],size = 1,fontFamily = '微软雅黑',color = "random-light",shape = 'circle',minRotation = -pi/4, maxRotation = pi/4, rotateRatio = 0.4)
vocab <- names(term.table)    #创建词库
get.terms <- function(x) {
  index <- match(x, vocab)  # 获取词的ID
  index <- index[!is.na(index)]  #去掉没有查到的，也就是去掉了的词
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))   #生成上图结构
}
documents <- lapply(doc.list, get.terms)
K <- 5   #主题数
G <- 5000    #迭代次数
alpha <- 0.10   
eta <- 0.02
library(LDAvis)
library(lda)

set.seed(357) 
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, num.iterations = G, alpha = alpha, eta = eta, initial = NULL, burnin = 0, compute.log.likelihood = TRUE)

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))  #文档—主题分布矩阵
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))  #主题-词语分布矩阵
term.frequency <- as.integer(term.table)   #词频
doc.length <- sapply(documents, function(x) sum(x[2, ]))
json <- createJSON(phi = phi, theta = theta, 
                   doc.length = doc.length, vocab = vocab,
                   term.frequency = term.frequency)
#json为作图需要数据，下面用servis生产html文件，通过out.dir设置保存位置

serVis(json = json, out.dir = 'D:/FTP/RData/vis', open.browser = FALSE)
writeLines(iconv(readLines("D:/FTP/RData/vis/lda.json"), from = "GBK", to = "UTF8"),
           file("D:/FTP/RData/vis/lda.json", encoding="UTF-8"))