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

training_accuracies = []
test_accuracies = []

for depth in range(31):
    tree = DecisionTreeClassifier(max_depth=depth)
    tree.fit(X_train, y_train)

    training_accuracies.append(tree.score(X_train, y_train))
    test_accuracies.append(tree.score(X_test, y_test))

print("training_accuracy = np.array(" + repr(training_accuracies) + ")")
print("test_accuracy = np.array(" + repr(test_accuracies) + ")")
