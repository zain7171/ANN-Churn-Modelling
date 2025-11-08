# Customer Churn Prediction using ANN and Streamlit

Predict whether a bank customer will leave using an **Artificial Neural Network (ANN)** with a **Streamlit web app**. Users can enter customer details to get real-time churn predictions.

---

## Features

- Preprocesses data including **categorical encoding** and **feature scaling**  
- Uses a **simple ANN** for binary classification (churn / stay)  
- Interactive **Streamlit interface** for real-time predictions

---

## How to Run

```bash
git clone https://github.com/zain7171/ANN-Churn-Modelling.git
cd ANN-Churn-Modelling
pip install -r requirements.txt
streamlit run streamlit_app.py
