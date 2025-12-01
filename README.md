# Spam Detection Project

## Overview

This project demonstrates a machine learning approach to email spam detection using the Naive Bayes algorithm. The model classifies SMS messages as either spam (unwanted messages) or ham (legitimate messages) with high accuracy.

**Dataset:** [Kaggle - Spam Emails Dataset](https://www.kaggle.com/datasets/abdallahwagih/spam-emails)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Usage](#usage)
- [Results](#results)

## Prerequisites

Ensure you have Python 3.7 or higher installed. The following libraries are required:

- pandas
- numpy
- scikit-learn

## Installation

Install all required packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

Alternatively, install packages individually:

```bash
pip install pandas numpy scikit-learn jupyter
```

## Dataset Description

The dataset contains 5,572 SMS messages with two columns:

- **Category**: Label indicating 'spam' or 'ham'
- **Message**: The actual text content of the message

**Distribution:**
- Ham messages: 4,825 (86.6%)
- Spam messages: 747 (13.4%)

## Model Architecture

The spam detection system uses the following approach:

1. **Text Vectorization**: CountVectorizer transforms text messages into numerical features
2. **Classification**: MultinomialNB (Naive Bayes) classifier for binary classification
3. **Training**: Model learns word frequency patterns associated with spam vs ham
4. **Evaluation**: Performance assessed on a held-out test set (25% of data)

## Performance Metrics

The model achieves excellent performance on the test dataset:

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | 0.99 | Overall correct classifications |
| **Precision** | 0.98 | Spam predictions that are actually spam |
| **Recall** | 0.94 | Percentage of spam messages detected |
| **F1 Score** | 0.96 | Balanced performance measure |

## Usage

### Running the Notebook

Open and run the Jupyter notebook:

```bash
jupyter notebook "Spam Detection.ipynb"
```

### Making Predictions

After training the model, you can classify new messages:

```python
# Example: Classifying a spam message
spam_email = ["Win $1,000,000 today! Click here now!"]
spam_vectorized = cv.transform(spam_email)
prediction = model.predict(spam_vectorized)
# Output: 1 (spam)

# Example: Classifying a legitimate message
ham_email = ["Can we schedule a meeting for tomorrow?"]
ham_vectorized = cv.transform(ham_email)
prediction = model.predict(ham_vectorized)
# Output: 0 (ham)
```

## Results

The Naive Bayes classifier successfully identifies spam messages with 99% accuracy. The model is particularly effective at:

- Detecting promotional language and offers
- Identifying urgent call-to-action phrases
- Recognizing spam-specific keywords (free, win, prize, call now)

**Key Insights:**
- Simple word frequency patterns are highly indicative of spam
- The model maintains high precision, minimizing false positives
- Excellent recall ensures most spam messages are caught

## Project Structure

```
spam_detection/
├── Spam Detection.ipynb    # Main analysis notebook with detailed comments
├── spam.csv                 # Dataset file
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Future Improvements

Potential enhancements to explore:

- Implement TF-IDF vectorization for better feature representation
- Try advanced models (SVM, Random Forest, Neural Networks)
- Add hyperparameter tuning with cross-validation
- Incorporate additional features (message length, special characters)
- Deploy as a web application or API

## License

This project is for educational purposes. Dataset credit goes to the original Kaggle contributor.

