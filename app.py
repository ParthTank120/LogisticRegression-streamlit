import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Title
st.title("Titanic Survival Prediction App")

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Data preprocessing
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train model
model = LogisticRegression()
model.fit(X, y)

# User Inputs
st.header("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 8, 0)
parch = st.number_input("Parents/Children aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)

# Convert inputs
sex = 0 if sex == "male" else 1

input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("The passenger is likely to Survive ✅")
    else:
        st.error("The passenger is Not likely to Survive ❌")
