import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import datasets, svm, metrics
import numpy as np
from sklearn.model_selection import train_test_split


# Importiere den Iris Datensatz
iris = datasets.load_iris()

# Benutze die ersten beiden Features
X = iris.data[:, :4]  # we only take the first two features.
Y = iris.target

# Train Test Split
# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=None)

# Erstelle Support Vector Classifier
kernel = "linear"
clf = svm.SVC(kernel=kernel, gamma='auto')

# Training
clf.fit(X_train, y_train)

# Predict the value
predicted = clf.predict(X_test)

# Score
score = clf.score(X_test, y_test)
print(f'score: {clf.score(X_test, y_test)}')

# Nach Aufgabe (f) entfernen
# quit()

# Erstelle plot
X0, X1 = X[:, 0], X[:, 1]                               # nur zwei features
# X0, X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]     # alle vier features
fig = plt.figure()
ax = fig.add_subplot()

""" disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.8,
    ax=ax,
    xlabel=iris.feature_names[0],
    ylabel=iris.feature_names[1],
) """

plt.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(f'SVC with : "{kernel}" kernel function and score : {score}')

plt.show()
