#test a classifier's accuracy
#not part of training data
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
test_idx = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx , axis = 0)

#testing data
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

#tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
#what tree predicts
print(clf.predict(test_data))


#viz code
print(test_data[1], test_target[1])

print(iris.feature_names, iris.target_names)