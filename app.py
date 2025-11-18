import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_predictor.pkl")
data = pd.read_csv("Salary Data.csv")
job_titles = sorted(data["Job Title"].dropna().unique())

st.title("Salary Prediction Tool")

age = st.number_input("Age", min_value=18, max_value=60, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
job_title = st.selectbox("Job Title", job_titles)
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

MIN_GAP = 16   

if st.button("Predict"):

    if age - experience < MIN_GAP:
        st.error(f"❌ Invalid Entry: Age - Difference between age and experience must be at least {MIN_GAP} years.")
    
    else:
        sample = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Education Level": education,
            "Job Title": job_title,
            "Years of Experience": experience
        }])

        predicted_salary = model.predict(sample)[0]
        st.success(f"Estimated Salary: ₹{predicted_salary:,.02f}")


