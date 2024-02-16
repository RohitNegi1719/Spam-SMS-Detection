import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the CSV file containing spam data
df = pd.read_csv('spam.csv', encoding='latin1', usecols=[0, 1])
df.rename(columns={'v1': 'Label', 'v2': 'Message'}, inplace=True)
df['Label'] = df['Label'].replace(['ham', 'spam'], [0, 1])

# Remove duplicate rows
df = df.drop_duplicates(keep='first')

# Define a function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    # Keep only alphanumeric characters and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Apply text preprocessing to the 'Message' column
df['Message'] = df['Message'].apply(preprocess_text)

# TF-IDF Vectorization with feature selection
max_features = 2000  # Increase the number of features
tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))  # Include bigrams
tfidf_features = tfidf_vectorizer.fit_transform(df['Message'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, df['Label'], test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Use Random Forest
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)
