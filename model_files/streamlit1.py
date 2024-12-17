import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# Load Models and Scalers
@st.cache_resource
def load_models():
    churn_model = load_model(r'model_files\churn_model.h5')
    price_model = load_model(r"model_files/book_price_prediction_model.h5")
    demand_model = load_model(r"model_files/demand_model.h5")

    with open(r"model_files/book_price_scaler.pkl", 'rb') as f:
        book_price_scaler = pickle.load(f)
    with open(r"model_files/demand_scaler.pkl", 'rb') as f:
        demand_scaler = pickle.load(f)
    with open(r"model_files/label_encoders.pkl", 'rb') as f:
        label_encoders = pickle.load(f)

    dataset = pd.read_csv(r"model_files/finalcleaned.csv", encoding='latin1')
    publishers = dataset['publisher_name'].dropna().unique().tolist()
    languages = dataset['language_name'].dropna().unique().tolist()

    return churn_model, price_model, demand_model, book_price_scaler, demand_scaler, label_encoders, publishers, languages


churn_model, price_model, demand_model, book_price_scaler, demand_scaler, label_encoders, publishers, languages = load_models()


# Utility Function for Safe Encoding
def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return -1


# Sidebar Navigation
st.sidebar.title("Prediction Models")
page = st.sidebar.radio("Choose a Prediction Task", ["Churn Prediction", "Book Price Prediction", "Demand Prediction"])

# 1. Churn Prediction
if page == "Churn Prediction":
    st.title("Customer Churn Prediction")
    st.write("Predict if a customer is Active or Inactive based on Recency, Frequency, and Monetary values.")

    recency = st.number_input("Recency (days since last purchase)", min_value=0.0, step=0.1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spending)", min_value=0.0, step=0.1)

    if st.button("Predict Churn"):
        input_data = np.array([[recency, frequency, monetary]])
        scaler = MinMaxScaler()
        input_data_scaled = scaler.fit_transform(input_data)

        churn_prediction = churn_model.predict(input_data_scaled)
        result = "Inactive Customer" if churn_prediction > 0.5 else "Active Customer"

        st.success(f"Prediction: {result}")

# 2. Book Price Prediction
elif page == "Book Price Prediction":
    st.title("Book Price Prediction")
    st.write("Predict the price of a book based on its details.")

    language = st.selectbox("Select Language", languages)
    publisher = st.selectbox("Select Publisher", publishers)
    city = st.text_input("City")
    country = st.text_input("Country")
    num_pages = st.number_input("Number of Pages", min_value=1, step=1)
    book_age = st.number_input("Book Age (in years)", min_value=0, step=1)

    if st.button("Predict Price"):
        try:
            # Encode categorical inputs
            language_encoded = safe_transform(label_encoders['language_name'], language)
            publisher_encoded = safe_transform(label_encoders['publisher_name'], publisher)
            city_encoded = safe_transform(label_encoders['city'], city)
            country_encoded = safe_transform(label_encoders['country_name'], country)

            # Prepare input DataFrame
            input_data = pd.DataFrame({
                'num_pages': [num_pages],
                'language_name_encoded': [language_encoded],
                'publisher_name_encoded': [publisher_encoded],
                'book_age': [book_age],
                'city_encoded': [city_encoded],
                'country_name_encoded': [country_encoded]
            })

            # Scale and predict
            input_data_scaled = book_price_scaler.transform(input_data)
            predicted_price = price_model.predict(input_data_scaled)[0][0]

            st.success(f"Predicted Price: ${round(predicted_price, 2)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# 3. Demand Prediction
elif page == "Demand Prediction":
    st.title("Book Demand Prediction")
    st.write("Predict the demand level for a book based on its details.")

    language = st.selectbox("Select Language", languages)
    publisher = st.selectbox("Select Publisher", publishers)
    city = st.text_input("City")
    country = st.text_input("Country")
    num_pages = st.number_input("Number of Pages", min_value=1, step=1)
    delivery_days = st.number_input("Delivery Days", min_value=1, step=1)

    if st.button("Predict Demand"):
        try:
            # Encode inputs
            language_encoded = safe_transform(label_encoders['language_name'], language)
            publisher_encoded = safe_transform(label_encoders['publisher_name'], publisher)
            city_encoded = safe_transform(label_encoders['city'], city)
            country_encoded = safe_transform(label_encoders['country_name'], country)

            # Prepare feature vector
            feature_vector = np.array([[num_pages, language_encoded, publisher_encoded, city_encoded, country_encoded, delivery_days]])
            feature_vector_scaled = demand_scaler.transform(feature_vector)

            # Predict demand
            predicted_demand = demand_model.predict(feature_vector_scaled)[0][0]
            if predicted_demand < 0.3:
                demand_label = "Low Demand"
            elif 0.3 <= predicted_demand < 0.7:
                demand_label = "Moderate Demand"
            else:
                demand_label = "High Demand"

            st.success(f"Predicted Demand: {demand_label} ({predicted_demand:.2f})")
        except Exception as e:
            st.error(f"Error: {str(e)}")
