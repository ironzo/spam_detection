
# Spam Detection Project

## Overview
This project aims to build a spam detection model using a dataset available on [Kaggle](https://www.kaggle.com/datasets/abdallahwagih/spam-emails). The model leverages machine learning techniques to distinguish between spam (unwanted or unsolicited emails) and ham (non-spam emails).

## Prerequisites
Before running the project, ensure you have the following libraries installed:
- Pandas
- NumPy
- scikit-learn

You can install these packages using pip:
```
pip install pandas numpy scikit-learn
```

## Data Inspection
The dataset consists of email messages classified as 'spam' or 'ham'. Here is a snippet of the data:
```
  Category                                            Message
0      ham  Go until jurong point, crazy.. Available only ...
1      ham                          Ok lar... Joking wif u oni...
2     spam  Free entry in 2 a wkly comp to win FA Cup fina...
3      ham  U dun say so early hor... U c already then say...
4      ham  Nah I don't think he goes to usf, he lives aro...
```

## Model Training and Evaluation
The model training process involves:
1. Splitting the data into training and test sets.
2. Transforming the text data into numerical data using a `CountVectorizer`.
3. Training a `MultinomialNB` (Naive Bayes) classifier.
4. Evaluating the model on test data.

## Testing the Model
The model can predict whether an email is spam or not based on its content. Here are the model's performance metrics on the test data:
- **Precision**: 0.96
- **Recall**: 0.93
- **Accuracy**: 0.98
- **F1 Score**: 0.95

## Example Usage
To classify a new email as spam or ham, you can pass it to the model as shown:
```python
# Example of predicting a spam email
spam_email = ["Win $1,000,000 today in our Free Casino! No registration needed. Click this link."]
model.predict(spam_email)

# Example of predicting a non-spam email
ham_email = ["Hello, please let me know when we can schedule a meeting?"]
model.predict(ham_email)
```

