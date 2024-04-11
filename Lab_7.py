from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train Random Forest classifier
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", accuracy)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, rf_pred)
print("Random Forest Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Print classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=data.target_names))
from sklearn.ensemble import AdaBoostClassifier

# Instantiate AdaBoost classifier
adaboost_model = AdaBoostClassifier(n_estimators=100, random_state=42)

# Train AdaBoost classifier
adaboost_model.fit(X_train, y_train)

# Make predictions
adaboost_pred = adaboost_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, adaboost_pred)
print("\nAdaBoost Accuracy:", accuracy)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, adaboost_pred)
print("AdaBoost Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('AdaBoost Confusion Matrix')
plt.show()

# Print classification report
print("\nAdaBoost Classification Report:")
print(classification_report(y_test, adaboost_pred, target_names=data.target_names))
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Define base estimators
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(random_state=42))
]

# Instantiate Stacking classifier with a logistic regression meta-model
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train Stacking classifier
stacking_model.fit(X_train, y_train)

# Make predictions
stacking_pred = stacking_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, stacking_pred)
print("\nStacking Accuracy:", accuracy)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, stacking_pred)
print("Stacking Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Stacking Confusion Matrix')
plt.show()

# Print classification report
print("\nStacking Classification Report:")
print(classification_report(y_test, stacking_pred, target_names=data.target_names))
