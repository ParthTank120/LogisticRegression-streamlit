import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

st.title("🚢 Titanic Survival Prediction")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Titanic-Dataset.csv")

# ---------------- PREPROCESSING ----------------
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df = df.dropna(subset=['Embarked'])

# Feature Engineering
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
        'Sex_male', 'Embarked_Q', 'Embarked_S']]

y = df['Survived']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------- RESULTS ----------------
st.subheader("📊 Model Performance")

accuracy = accuracy_score(y_test, y_pred)
st.write("### Accuracy:", round(accuracy, 3))

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# ---------------- CONFUSION MATRIX ----------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)
