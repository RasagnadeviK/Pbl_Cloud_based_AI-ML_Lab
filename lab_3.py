from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt

# Load breast cancer dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build decision tree model using Gini criterion
model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)

# Predict on the test set
expected = y_test
predicted = model.predict(X_test)

# Print confusion matrix
print("Confusion Matrix:")
print(metrics.confusion_matrix(expected, predicted))

# Print accuracy
accuracy = metrics.accuracy_score(expected, predicted)
print("Accuracy:", accuracy)

# Plot decision tree function graph
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()
