from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Predict on the test set
predicted = model.predict(X_test)

# Print confusion matrix and accuracy
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, predicted))
accuracy = metrics.accuracy_score(y_test, predicted)
print("Accuracy:", accuracy)
import matplotlib.pyplot as plt

# Get feature importances
importances = model.feature_importances_

# Get feature names
feature_names = iris.feature_names

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Extract a single decision tree from the random forest
estimator = model.estimators_[0]

# Plot the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Extract a single decision tree from the random forest
estimator = model.estimators_[0]

# Plot the decision tree with adjusted box size
plt.figure(figsize=(40, 10))
plot_tree(estimator, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, fontsize=8)
plt.show()

