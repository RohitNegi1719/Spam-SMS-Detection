# Spam Classification using Random Forest

## Overview
This project aims to classify SMS messages as spam or non-spam (ham) using a Random Forest classifier. The classification is based on TF-IDF vectorization of the text data, which converts each message into a numerical representation. Random Forest is then used to train a model on this representation to predict whether a message is spam or not.

## Dependencies
- pandas
- numpy
- re
- nltk
- scikit-learn

## Dataset
The dataset used for this project is a collection of SMS messages labeled as spam or ham. It contains two columns: 'Label' indicating whether the message is spam (1) or ham (0), and 'Message' containing the text of the SMS.

## Workflow
1. **Data Preprocessing**: The dataset is read from a CSV file and preprocessed. Preprocessing steps include converting labels to numerical format (0 for ham, 1 for spam), removing duplicate rows, and cleaning the text data (lowercasing, removing non-alphanumeric characters, and stopwords).
2. **Feature Extraction**: TF-IDF vectorization is used to convert text data into numerical features. Bigrams (two-word combinations) are also included to capture more context.
3. **Model Training**: The data is split into training and testing sets. A Random Forest classifier is trained on the training data.
4. **Model Evaluation**: The trained model is evaluated on the testing set using accuracy score, classification report, and confusion matrix.

## Usage
1. Ensure you have the required dependencies installed.
2. Download the dataset (`spam.csv`) and save it in the same directory as your script.
3. Run the script to train the model and evaluate its performance.
4. Optionally, you can deploy the trained model for real-time predictions.

## Results
The model achieved an accuracy of X% on the test data. The classification report provides detailed metrics such as precision, recall, and F1-score for each class (spam and ham). The confusion matrix visualizes the model's performance in terms of true positives, true negatives, false positives, and false negatives.

## Author
Rohit Negi

## Acknowledgments
- Thanks to the authors of the dataset used in this project.
