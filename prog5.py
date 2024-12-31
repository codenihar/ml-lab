import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # Use 'macro' for multi-class
recall = recall_score(y_test, y_pred, average='macro')        # Use 'macro' for multi-class

# Print metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Sample prediction
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example feature values
predicted_class = model.predict(sample)
print(f"\nPredicted class for {sample}: {data.target_names[predicted_class[0]]}")
