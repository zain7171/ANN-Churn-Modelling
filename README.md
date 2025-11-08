Customer Churn Prediction using ANN and Streamlit

Predict whether a bank customer will leave using an Artificial Neural Network (ANN) with a Streamlit web app. Users can enter customer details to get real-time churn predictions.

How to Run
git clone https://github.com/zain7171/ANN-Churn-Modelling.git
cd ANN-Churn_Modelling
pip install -r requirements.txt
streamlit run streamlit_app.py

Dataset
Churn_Modelling_Sample.csv – features include CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary; target is Exited (1 = churn, 0 = stay).

Project Structure
streamlit_app.py
Churn_Modelling_Sample.csv
requirements.txt

Tech Stack

Python, TensorFlow/Keras, Scikit-learn, Streamlit, NumPy, Pandas