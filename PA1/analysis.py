import numpy as np
from decision_tree import DecisionTreeClassifier
import draw_tree

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

tree = DecisionTreeClassifier(max_depth=None)
tree.fit(X_train, y_train)

print("Training accuracy:", tree.score(X_train, y_train))
print("Test accuracy:", tree.score(X_test, y_test))
print("Feature importances:", tree.feature_importances_)
print("Tree depth:", tree.get_depth())

predictions = tree.predict(X_test)
draw_tree.draw_tree(X_train, y_train, tree)