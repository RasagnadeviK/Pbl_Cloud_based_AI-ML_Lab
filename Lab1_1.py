import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Creating a sample dataset
data = {
    'email': [
        "Get Viagra for free now!",
        "Meeting tomorrow at 10am.",
        "Make money fast, guaranteed!",
        "Reminder: Your appointment is tomorrow.",
        "Enlarge your assets with our product!",
        "Don't forget to submit your assignment.",
        "Claim your prize now!",
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam']
}

# Creating DataFrame
df = pd.DataFrame(data)

# Splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.2, random_state=42)

# Vectorizing the emails
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Training Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vect, y_train)

# Making predictions
predictions = clf.predict(X_test_vect)

# Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# New email
new_email = ["Win a free vacation now!"]

# Vectorizing the new email
new_email_vect = vectorizer.transform(new_email)

# Making prediction
prediction_new = clf.predict(new_email_vect)
print("Prediction for new email:", prediction_new[0])
