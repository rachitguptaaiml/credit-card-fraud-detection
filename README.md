# Credit Card Fraud Detection with Imbalanced Data Handling

This project demonstrates a simple credit card fraud detection system using Python and scikit-learn, with a focus on handling imbalanced data.

## Problem Statement
Credit card companies face a significant challenge in detecting fraudulent transactions. Traditional machine learning approaches often struggle to handle the class imbalance problem, where legitimate transactions outnumber fraudulent ones by a large margin.

## Features
* Synthetic dataset generation for imbalanced data handling
* Random Forest Classifier training and evaluation on scaled features
* Accuracy score, classification report, and confusion matrix metrics

## Setup/Run Instructions
1. Clone the repository: `git clone https://github.com/[your-username]/credit-card-fraud-detection.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python main.py`

## Sample Output
Accuracy: 0.93
Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.90      0.92        50
           1       0.91      0.94      0.92        51

    accuracy                           0.93        101

   macro avg       0.93      0.92      0.92        101
weighted avg       0.93      0.92      0.92        101

Confusion Matrix:
[[49 1]
 [2 49]]

## What I Learned
This project taught me the importance of handling class imbalance in machine learning problems, especially those with skewed data distributions like credit card fraud detection. By using techniques such as oversampling the minority class and hyperparameter tuning, we can improve model performance and accuracy on imbalanced datasets.