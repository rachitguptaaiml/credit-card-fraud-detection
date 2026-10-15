import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate synthetic data for imbalanced dataset
np.random.seed(42)
df = pd.DataFrame({
    'Amount': np.random.uniform(10, 1000, 10000),
    'Time': np.random.uniform(1, 24 * 60, 10000),
    'Location': np.random.choice(['USA', 'Canada', 'UK'], 10000),
    'Card Type': np.random.choice(['Visa', 'Mastercard', 'Amex'], 10000),
    'Fraud': np.random.choice([0, 1], 10000)
})

# Define features and target
X = df.drop('Fraud', axis=1)
y = df['Fraud']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier on the scaled data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test_scaled)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:')
print(report)
print('Confusion Matrix:')
print(matrix)