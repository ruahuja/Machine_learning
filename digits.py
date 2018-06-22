#!usr/bin/python3
from sklearn.datasets import load_digits
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

digit=load_digits()
plt.imshow(digit.images[0])
plt.show()
digit.images

#loading digit images
digit=load_digits()

#only feature data
training_data=digit.data

#only target data
training_target=digit.target

#training data extract from original data
td_original=np.delete(training_data,-1,axis=0)

#training target extract from original data
tt_original=np.delete(training_target,-1)

#calling support vector classifier
clf=SVC()

#tarining algo
trained=clf.fit(td_original,tt_original)

#now time for prediction
output=trained.predict(digit.data[-1].reshape(1,64))
print(output)

#plotting that testing image
plt.imshow(digit.images[-1])
plt.show()

