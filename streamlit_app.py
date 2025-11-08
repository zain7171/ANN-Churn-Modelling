import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load the Dataset and Pre-process
dataset = pd.read_csv('Churn_Modelling_Sample.csv')
features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
X = dataset[features].values
y = dataset['Exited'].values

# Encode Gender
le = LabelEncoder()
gender_id = features.index('Gender')
X[:, gender_id] = le.fit_transform(X[:, gender_id])

# Encode Geography
geo_id = features.index('Geography')
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [geo_id])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Build and Train ANN
model = Sequential()
model.add(Dense(units=16, activation='relu', input_dim=X.shape[1]))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=50, verbose=0)

# Streamlit UI
st.title("Customer Churn Predictor")
st.write("Predict if a customer will leave the bank.")

# Inputs
credit_score = st.number_input("Credit Score", 300, 900, 650)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18,100,35)
tenure = st.number_input("Tenure(years)", 0,10,3)
balance = st.number_input("Balance", 0.0,250000.0,100000.0)
num_of_products = st.number_input("Number of Products", 1,4,2)
has_cr_card = st.selectbox("Has a Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

if st.button("Predict Churn"):
    x = np.array([[credit_score, geography, gender, age, tenure, balance,
                   num_of_products, has_cr_card, is_active_member, estimated_salary]])
    x[:, gender_id] = (x[:, gender_id] == 'Male').astype(int)
    x[:, features.index('HasCrCard')] = (x[:, features.index('HasCrCard')] == 'Yes').astype(int)
    x[:, features.index('IsActiveMember')] = (x[:, features.index('IsActiveMember')] == 'Yes').astype(int)
    x = np.array(ct.transform(x))
    x = scaler.transform(x)
    prob = model.predict(x)[0][0]
    churn = prob > 0.5
    st.metric("Churn Probability", f"{prob*100:.2f}%")
    if churn:
        st.warning("Prediction: Customer WILL leave.")
    else:
        st.success("Prediction: Customer will stay.")