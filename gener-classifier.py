import numpy
from sklearn import gaussian_process, metrics, neighbors, tree

clf = tree.DecisionTreeClassifier()
clf2 = neighbors.KNeighborsClassifier()
clf3 = gaussian_process.GaussianProcessClassifier()


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf = clf.fit(X, Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X, Y)

prediction = clf.predict(X)
acc = metrics.accuracy_score(Y, prediction) * 100
prediction2 = clf2.predict(X)
acc2 = metrics.accuracy_score(Y, prediction2) * 100
prediction3 = clf3.predict(X)
acc3 = metrics.accuracy_score(Y, prediction3) * 100


print("Decision Tree: ", prediction, " Accuracy: ", acc)
print("Nearest Neighbor: ", prediction2, " Accuracy: ", acc2)
print("Guassian Process: ", prediction3, " Accuracy: ", acc3)

index = numpy.argmax([acc, acc2, acc3])
print('\n', index)
classifiers = {0:'Decision Tree', 1:'Nearest Neighbor', 2:'Guassian Process'}
print('Best gender classifier is: {}'.format(classifiers[index]))
