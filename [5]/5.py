import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the dataset from CSV file
data = pd.read_csv('Question5_Multi_Class_Dataset.csv')

# Prepare the features (X) and target (y)
X = data[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']].values  # 5 input features
y = data['Target'].values  # Output variable (0, 1, or 2)

# Split the dataset: 70% training, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression classifier
clf_lr = LogisticRegression(random_state=42, max_iter=1000)
clf_lr.fit(X_train, y_train)

# Predict on the test set
y_pred = clf_lr.predict(X_test)

# Compute metrics for each class (0, 1, 2)
accuracy = accuracy_score(y_test, y_pred)
f1_class_0 = f1_score(y_test, y_pred, average=None, labels=[0])[0]  # F1 for class 0
f1_class_1 = f1_score(y_test, y_pred, average=None, labels=[1])[0]  # F1 for class 1
f1_class_2 = f1_score(y_test, y_pred, average=None, labels=[2])[0]  # F1 for class 2

# Round all metrics to 3 decimal places
accuracy_rounded = round(accuracy, 3)
f1_class_0_rounded = round(f1_class_0, 3)
f1_class_1_rounded = round(f1_class_1, 3)
f1_class_2_rounded = round(f1_class_2, 3)

# Print results for verification
print(f"Accuracy: {accuracy_rounded}")
print(f"F-1 score (class = 0): {f1_class_0_rounded}")
print(f"F-1 score (class = 1): {f1_class_1_rounded}")
print(f"F-1 score (class = 2): {f1_class_2_rounded}")

# Convert metrics to strings for digit-by-digit output (for drag-and-drop)
def digits_to_list(number):
    number_str = f"{number:.3f}"  # Format to 3 decimal places
    return [int(d) for d in number_str.replace('.', '')]  # Remove decimal and convert to list of integers

# Get digits for each metric
accuracy_digits = digits_to_list(accuracy_rounded)
f1_class_0_digits = digits_to_list(f1_class_0_rounded)
f1_class_1_digits = digits_to_list(f1_class_1_rounded)
f1_class_2_digits = digits_to_list(f1_class_2_rounded)

# Print digits for drag-and-drop (each metric should have 5 digits: X.XXX → [X,X,X,X,X])
print("\nDigits for Accuracy:", accuracy_digits)
print("Digits for F-1 score (class = 0):", f1_class_0_digits)
print("Digits for F-1 score (class = 1):", f1_class_1_digits)
print("Digits for F-1 score (class = 2):", f1_class_2_digits)