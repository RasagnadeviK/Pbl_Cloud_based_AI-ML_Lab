import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

data = {
    'Pclass': [1, 2, 3, 1, 3],
    'Sex': ['female', 'female', 'male', 'male', 'female'],
    'Age': [29, 35, 24, 40, np.nan],
    'SibSp': [1, 1, 0, 1, 0],
    'Parch': [2, 2, 0, 1, 0],
    'Fare': [100, 50, 10, 80, 15],
    'Embarked': ['S', 'C', 'S', 'S', 'C'],
    'Survived': [1, 1, 0, 1, 0]
}

titanic_df = pd.DataFrame(data)
titanic_df.to_csv('my_titanic.csv', index=False)
print("Titanic dataset created and saved as 'my_titanic.csv'")

titanic_data = pd.read_csv("my_titanic.csv")
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using entropy instead of the default "gini"
clf = DecisionTreeClassifier(criterion='entropy')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.show()
# New dataset
new_data = {
    'Pclass': [1, 2, 3, 1, 2],
    'Sex': ['male', 'female', 'female', 'male', 'male'],
    'Age': [25, 45, 30, 28, 50],
    'SibSp': [1, 0, 1, 0, 1],
    'Parch': [1, 2, 0, 1, 2],
    'Fare': [90, 60, 20, 100, 70],
    'Embarked': ['C', 'S', 'S', 'C', 'S'],
    'Survived': [1, 0, 1, 0, 1]  # Just a placeholder for the new data, actual survival is unknown
}

new_titanic_df = pd.DataFrame(new_data)
new_X = new_titanic_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
new_y = new_titanic_df['Survived']

# Convert categorical variables to numerical
new_X['Sex'] = new_X['Sex'].map({'male': 0, 'female': 1})
new_X['Embarked'] = new_X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Predict using the trained classifier
new_y_pred = clf.predict(new_X)

# Calculate accuracy
new_accuracy = accuracy_score(new_y, new_y_pred)
print("New Data Accuracy:", new_accuracy)

# Confusion matrix
new_conf_matrix = confusion_matrix(new_y, new_y_pred)
print("New Data Confusion Matrix:")
print(new_conf_matrix)

# Classification report
new_class_report = classification_report(new_y, new_y_pred)
print("New Data Classification Report:")
print(new_class_report)

# Plot decision tree for the new dataset
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=new_X.columns, class_names=['Not Survived', 'Survived'])
plt.show()
