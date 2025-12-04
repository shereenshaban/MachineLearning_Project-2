import streamlit as st
import numpy as np
import pickle

st.title("Titanic Survival Prediction")

@st.cache_resource
def load_model():
    with open(r'E:\projects\ML_Titanc\titanic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

with st.form("passenger_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", options=[1, 2, 3],help="1 = 1st, 2 = 2nd, 3 = 3rd")
        sex= st.selectbox("Sex", options=["male", "female"])
        age = st.number_input("Age", min_value=0, max_value=100, value=25)

    with col2:
        sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2) 
        embarked = st.selectbox("Port of Embarkation", options=["C", "Q", "S"], help="C = Cherbourg, Q = Queenstown, S = Southampton")    
    submit_button = st.form_submit_button("Predict Survival")    

if submit_button:
    sex_encoded = 1 if sex == "male" else 0
    embarked_encoded = {"C": 0, "Q": 1, "S": 2}
    feature=[[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]]

    prediction = model.predict(feature)[0]
    prediction_proba = model.predict_proba(feature)[0][prediction]
    if prediction == 1:
        st.success(f"Survived with a probability of {prediction_proba:.2f}.")
        st.balloons()
    else:
        st.error(f"Did not survive with a probability of {prediction_proba:.2f}.")
        
    