import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocess import preprocess_data


# Load processed data
X_train, X_test, y_train, y_test = preprocess_data()

# Load trained model
model = joblib.load(
    '../models/gender_model.pkl'
)

# Predictions
predictions = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average='weighted'
)

recall = recall_score(
    y_test,
    predictions,
    average='weighted'
)

f1 = f1_score(
    y_test,
    predictions,
    average='weighted'
)

# Print metrics
print('Accuracy:', accuracy)

print('Precision:', precision)

print('Recall:', recall)

print('F1 Score:', f1)

# Classification Report
print('\\nClassification Report:\\n')

print(classification_report(y_test, predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

# Plot confusion matrix
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix')

plt.show()
