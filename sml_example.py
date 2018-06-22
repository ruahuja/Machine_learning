#!/usr/bin/python3

from sklearn import tree

#features of apple and orange

data=[[100,0],[130,0],[135,1],[150,1]]
output=["apple","apple","orange","orange"]

#decision tree algo call

algo=tree.DecisionTreeClassifier()

#train data
trained_algo=algo.fit(data,output)

#now teesting phase

predict=trained_algo.predict([[100,0]])

print(predict)
