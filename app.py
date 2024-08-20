# app.py
import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler



model = joblib.load('customer_conversion_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Digital Marketing Campaign Conversion Prediction")

st.write("""
This application predicts the likelihood of a customer converting based on demographic and engagement factors. 
Fill in the details below to get a prediction.
""")

age = st.number_input("Age")
income = st.number_input("Income")
ad_spend = st.number_input("Ad Spend")
click_through_rate = st.number_input("Click Through Rate")
conversion_rate = st.number_input("Conversion Rate")
website_visits = st.number_input("Website Visits")
pages_per_visit = st.number_input("Pages Per Visit")
time_on_site = st.number_input("Time On Site (minutes)")
social_shares = st.number_input("Social Shares")
email_opens = st.number_input("Email Opens")
email_clicks = st.number_input("Email Clicks")
previous_purchases = st.number_input("Previous Purchases")
loyalty_points = st.number_input("Loyalty Points")

input_features = np.array([[age, income, ad_spend, click_through_rate, conversion_rate,
                            website_visits, pages_per_visit, time_on_site, social_shares,
                            email_opens, email_clicks, previous_purchases, loyalty_points]])

input_features = scaler.transform(input_features)


if st.button("Predict"):
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features) 
    st.write(f"Predicted Probability of Conversion: {prediction_proba[0][1]:.2f}")
    
    threshold = 0.5
    
    if prediction_proba[0][1] > threshold:
        st.success("This customer is likely to convert!")
    else:
        st.warning("This customer is unlikely to convert.")



