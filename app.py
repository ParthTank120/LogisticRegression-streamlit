import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- TITLE ----------------
st.title("🚢 Titanic Survival Prediction with Confusion Matrix")

# ---------------- LOAD DATA ----------------
data = sns.load_dataset("titanic")

data = data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'survived']]
data['age'].fillna(data['age'].mean(), inplace=True)
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

X = data.drop('survived', axis=1)
y = data['survived']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("### Model Accuracy:", round(accuracy * 100, 2), "%")

# ---------------- CONFUSION MATRIX ----------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)

# ---------------- USER INPUT ----------------
st.header("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 8, 0)
parch = st.number_input("Parents/Children aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)

sex = 0 if sex == "male" else 1

if st.button("Predict Survival"):

    input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ The passenger is likely to SURVIVE")
    else:
        st.error("❌ The passenger is NOT likely to survive")
