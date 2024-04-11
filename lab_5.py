import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Read the Fish dataset
dataset_url = "https://raw.githubusercontent.com/harika-bonthu/SupportVectorClassifier/main/datasets_229906_491820_Fish.csv"
fish = pd.read_csv(dataset_url)

# Define features and target variable
X = fish.drop(['Species'], axis='columns')
y = fish['Species']

# Empty lists to store accuracy values and confusion matrices
accuracies = []
conf_matrices = []

# Number of experiments
num_experiments = 10

# Perform multiple experiments
for _ in range(num_experiments):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Instantiate the SVC object with a linear kernel and regularization parameter C=1
    model = SVC(kernel='linear', C=1)

    # Train the SVC classifier using the training data
    model.fit(X_train, y_train)

    # Make predictions
    svm_pred = model.predict(X_test)

    # Calculate accuracy and store it
    accuracy = model.score(X_test, y_test)
    accuracies.append(accuracy)

    # Print accuracy
    print(f"Accuracy for Experiment {_ + 1}: {accuracy}")

    # Calculate confusion matrix and store it
    conf_matrix = confusion_matrix(y_test, svm_pred)
    conf_matrices.append(conf_matrix)

    # Print classification report for each experiment
    print(f"\nClassification Report for Experiment {_ + 1}:")
    print(classification_report(y_test, svm_pred))

# Plot line graph of accuracy over experiments
plt.plot(range(1, num_experiments + 1), accuracies, marker='o')
plt.xlabel('Experiment')
plt.ylabel('Accuracy')
plt.title('Accuracy over Experiments')
plt.xticks(range(1, num_experiments + 1))
plt.show()
