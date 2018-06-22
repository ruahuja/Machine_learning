#!usr/bin/python3

import numpy
from sklearn.datasets import load_iris
from sklearn import tree

#loading all data
iris=load_iris()

#print feature names
print(iris.feature_names)


#print target name
print(iris.target_names)

#print training data
#print(iris.data)
#only setosa
setosa=iris.data[0:50]

#print target data
#print(iris.target)
#only setosa
s_data=iris.target[0:50]
print(s_data)
print(s_data.size)

x=[0,50,100]
only_target_training=numpy.delete(iris.target,x)
print(only_target_training)
print(only_target_training.size)

#testing target
test_target=iris.target[x]
print(test_target)

#train data
only_data_train=numpy.delete(iris.data,x,axis=0)
print(only_data_train)

#test data
test_data=iris.data[x]
print(test_data)

#calling algo

clf=tree.DecisionTreeClassifier()
trained=clf.fit(only_data_train,only_target_training)
output=trained.predict(test_data)
print(output)

