#!user/bin/python3
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import numpy
import matplotlib.pyplot as plt
#loading iris datasets

iris=load_iris()

#trainig flowers features stored in iris.data
#output accordingly stored in iris.target

#now splitting into test and train sets

train_iris, test_iris, train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.2)

#calling decisiontree classifier
dsclf=tree.DecisionTreeClassifier()

#calling KNN algo
knnclf=KNeighborsClassifier(n_neighbors=3)

#training data
traineddsc=dsclf.fit(train_iris,train_target)
trainedknn=knnclf.fit(train_iris,train_target)

#testing algo
outputdsc=traineddsc.predict(test_iris)
print(outputdsc)
outputknn=trainedknn.predict(test_iris)
print(outputknn)

#orignal output
print(test_target)

#calculating accuracy decisiontree
pctd=accuracy_score(test_target,outputdsc)
print(pctd)

#calculating accuracy for knn
pct=accuracy_score(test_target,outputknn)
print(pct)

export_graphviz(dsclf, out_file="tree.dot", max_depth=8,feature_names=iris.feature_names, class_names=None, label='all', filled=True, node_ids=True)

plt.bar(outputdsc,outputknn,label="output")
plt.bar(pctd,pct,label="accuracy")
plt.legend()
plt.show()

