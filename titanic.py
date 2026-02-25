# Titanic Survival Prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
data = pd.read_csv("titanic.csv")

# Select Required Columns
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

# Handle Missing Values
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Convert Categorical to Numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Features and Target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ----------- Custom Prediction -----------
print("\n--- Test with New Passenger Data ---")

# Example passenger
# Pclass, Sex (0=male,1=female), Age, SibSp, Parch, Fare
new_passenger = np.array([[3, 0, 22, 1, 0, 7.25]])

prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("Passenger is likely to Survive")
else:
    print("Passenger is Not likely to Survive")