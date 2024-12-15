from django.shortcuts import render
from django.views import View
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import load_model
import os
from django.http import JsonResponse
import pickle
from django.http import HttpResponse

# Load models
churn_model = load_model(r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\completed merge\churn_model.h5")
predict_price_model = load_model(r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\completed merge\book_price_prediction_model.h5")
        

# Function to predict churn
def predict_churn(recency, frequency, monetary):
    scaler = MinMaxScaler()
    input_data = np.array([[recency, frequency, monetary]])
    input_data_scaled = scaler.fit_transform(input_data)
    churn_prediction = churn_model.predict(input_data)
    return 'Inactive customer' if churn_prediction > 0.5 else 'Active customer'



# View for the main page
class IndexView(View):
    def get(self, request):
        return render(request, 'urldb/index.html')


# Churn Prediction View
class ChurnPredictionView(View):
    def get(self, request):
        return render(request, 'urldb/churn_prediction.html')

    def post(self, request):
        recency = float(request.POST['recency'])
        frequency = int(request.POST['frequency'])
        monetary = float(request.POST['monetary'])
        churn_result = predict_churn(recency, frequency, monetary)
        return render(request, 'urldb/churn_prediction.html', {'churn_result': churn_result})





# Vectorizer should be initialized once during application startup
# vectorizer = TfidfVectorizer(max_features=5000)
# vectorizer.fit(books_df['title'])  # Fit the vectorizer once



# def safe_transform(encoder, value):
#     try:
#         return encoder.transform([value])[0]
#     except ValueError:  # In case of unseen labels
#         return -1  # Return a default value like -1 for unseen labels

# class PricePredictionView(View):
#     def __init__(self):
#         # Load the pre-trained model and scaler
#         self.price_model = load_model(r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\merged\price_model.h5")
        
#         # Load the scaler
#         with open(r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\merged\price_scaler.pkl", 'rb') as f:
#             self.price_scaler = pickle.load(f)
        
#         # Load label encoders for categorical columns
#         self.label_encoders = pickle.load(open(r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\merged\label_encoders.pkl", 'rb'))

#     def get(self, request):
#         # Load the dataset to get the unique publishers and languages
#         dataset_path = r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\completed merge\finalcleaned.xlsx"
#         df = pd.read_excel(dataset_path)

#         # Get the unique publishers and languages from the dataset
#         publishers = df['publisher_name'].dropna().unique().tolist()
#         languages = df['language_name'].dropna().unique().tolist()

#         # Render the form with the list of publishers and languages
#         return render(request, 'urldb/predict_price.html', {'publishers': publishers, 'languages': languages})

#     def post(self, request):
#         try:
#             # Load the dataset to get the unique publishers and languages again
#             dataset_path = r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\completed merge\finalcleaned.xlsx"
#             df = pd.read_excel(dataset_path)

#             # Get the unique publishers and languages from the dataset
#             publishers = df['publisher_name'].dropna().unique().tolist()
#             languages = df['language_name'].dropna().unique().tolist()

#             # Get data from the request body (POST form submission)
#             data = request.POST

#             # Extract relevant features from the form data
#             language_name = data.get('language_name')  # Language as selection
#             publisher_name = data.get('publisher_name')
#             city = data.get('city')
#             country=data.get('country_name')
#             num_pages = int(data.get('num_pages'))

#             # Encode categorical variables using the label encoder
#             language_encoded = safe_transform(self.label_encoders['language_name'], language_name)
#             publisher_encoded = safe_transform(self.label_encoders['publisher_name'], publisher_name)
#             city_encoded = safe_transform(self.label_encoders['city'], city)
#             country_encoded=safe_transform(self.label_encoders['country_name'], country)

#             # Prepare the feature vector
#             feature_vector = np.array([[num_pages, language_encoded, publisher_encoded, city_encoded,country_encoded]])

#             # Scale the features using the scaler
#             feature_vector_scaled = self.price_scaler.transform(feature_vector)

#             # Predict the price using the trained model
#             predicted_price = self.price_model.predict(feature_vector_scaled)[0][0]

#             # Return the prediction along with the list of publishers and languages for the form
#             return render(request, 'urldb/predict_price.html', {
#                 'predicted_price': predicted_price,
#                 'publishers': publishers,
#                 'languages': languages
            
#             })

#         except Exception as e:
#             # In case of any error, return a JSON response with the error message
#             return JsonResponse({'error': str(e)}, status=400)

# Function to safely encode labels
def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:  # Handle unseen labels
        return -1  # Default fallback for unseen labels

class PricePredictionView(View):
    def __init__(self):
        # Load the pre-trained model
        # self.model = load_model(r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\completed merge\book_price_prediction_model.h5")
        print('_____a')

        # Load the saved scaler
        with open(r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\merged\book_price_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        print('_____b')
        # Load pre-fitted label encoders
        with open(r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\merged\label_encoders.pkl", 'rb') as f:
            encoders = pickle.load(f)
            self.label_encoder_language = encoders['language_name']
            self.label_encoder_publisher = encoders['publisher_name']
            self.label_encoder_city = encoders['city']
            self.label_encoder_country = encoders['country_name']
        print('_____c')

        # Load dataset once for publishers and languages
        dataset_path = r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\completed merge\finalcleaned.csv"
        df = pd.read_csv(dataset_path,encoding='latin1')
        print('_____d')

        self.publishers = df['publisher_name'].dropna().unique().tolist()
        self.languages = df['language_name'].dropna().unique().tolist()

    def get(self, request):
        # Render the form with the list of publishers and languages
        return render(request, 'urldb/predict_price.html', {
            'publishers': self.publishers,
            'languages': self.languages
        })

    def post(self, request):
        try:
            # Extract data from POST request
            print('____1')
            data = request.POST
            # Validate inputs
            language_name = data.get('language_name')
            publisher_name = data.get('publisher_name')
            city = data.get('city')
            country = data.get('country_name')
            num_pages = data.get('num_pages')
            book_age = data.get('book_age')

            # Check for missing values
            if not all([language_name, publisher_name, city, country, num_pages, book_age]):
                raise ValueError("All fields are required.")
            print('____2')

            # Convert numeric fields
            num_pages = int(num_pages)
            book_age = int(book_age)

            # Encode categorical variables
            language_encoded = safe_transform(self.label_encoder_language, language_name)
            publisher_encoded = safe_transform(self.label_encoder_publisher, publisher_name)
            city_encoded = safe_transform(self.label_encoder_city, city)
            country_encoded = safe_transform(self.label_encoder_country, country)
            print('____3')

            # Create a DataFrame for new input data
            new_data = pd.DataFrame({
                'num_pages': [num_pages],
                'language_name_encoded': [language_encoded],
                'publisher_name_encoded': [publisher_encoded],
                'book_age': [book_age],
                'city_encoded': [city_encoded],
                'country_name_encoded': [country_encoded]
            })
            print('____4')

            # Scale the data
            new_data_scaled = self.scaler.transform(new_data)
            print('____5')

            # Make predictions
            predicted_price = predict_price_model.predict(new_data_scaled)
            print('____6')

            # Return the prediction and form inputs
            return render(request, 'urldb/predict_price.html', {
                'predicted_price': round(predicted_price[0][0], 2),  # Format output
                'publishers': self.publishers,
                'languages': self.languages
                # 'language_name': language_name,
                # 'publisher_name': publisher_name,
                # 'city': city,
                # 'country_name': country,
                # 'num_pages': num_pages,
                # 'book_age': book_age
            })
        except Exception as e:
            # Handle errors gracefully
            return JsonResponse({'error': str(e)}, status=400)




class DemandPredictionView(View):
    def __init__(self):
        # Load the pre-trained model and scaler for demand prediction
        self.demand_model = load_model(r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\merged\demand_model.h5")
        
        # Load the scaler for demand prediction
        with open(r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\merged\demand_scaler.pkl", 'rb') as f:
            self.demand_scaler = pickle.load(f)
        
        # Load label encoders for categorical columns
        with open(r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\merged\label_encoders.pkl", 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        # Load the dataset to get the unique publishers and languages
        dataset_path = r"C:\Users\NAGARAJAN K\Desktop\guvi_final_project\completed merge\finalcleaned.csv"
        self.df = pd.read_csv(dataset_path,encoding='latin1')

        # Get the unique publishers and languages from the dataset
        self.publishers = self.df['publisher_name'].dropna().unique().tolist()
        self.languages = self.df['language_name'].dropna().unique().tolist()

    def safe_transform(self, encoder, value):
        """
        Safely transform a value using the label encoder, 
        handling unknown categories by returning a default value (e.g., -1).
        """
        try:
            return encoder.transform([value])[0]
        except ValueError:  # In case of unseen labels
            return -1  # Return a default value like -1 for unseen labels

    def get(self, request):
        # Render the form with the list of publishers and languages
        return render(request, 'urldb/predict_demand.html', {'publishers': self.publishers, 'languages': self.languages})

    def post(self, request):
        try:
            # Extract data from the form submission
            data = request.POST

            language_name = data.get('language_name')
            publisher_name = data.get('publisher_name')
            city = data.get('city')
            country = data.get('country_name')
            num_pages = int(data.get('num_pages'))
            delivery_days = int(data.get('delivery_days'))

            # Encode the categorical variables using the label encoder
            language_encoded = self.safe_transform(self.label_encoders['language_name'], language_name)
            publisher_encoded = self.safe_transform(self.label_encoders['publisher_name'], publisher_name)
            city_encoded = self.safe_transform(self.label_encoders['city'], city)
            country_encoded = self.safe_transform(self.label_encoders['country_name'], country)

            # Prepare the feature vector for demand prediction
            feature_vector = np.array([[num_pages, language_encoded, publisher_encoded, city_encoded, country_encoded, delivery_days]])

            # Scale the features using the scaler for demand prediction
            feature_vector_scaled = self.demand_scaler.transform(feature_vector)

            # Predict the demand using the trained model
            predicted_demand = self.demand_model.predict(feature_vector_scaled)[0][0]

            # Interpret the demand level
            if predicted_demand < 0.3:
                demand_label = "Low demand"
            elif 0.3 <= predicted_demand < 0.7:
                demand_label = "Moderate demand"
            else:
                demand_label = "High demand"

            # Return the prediction along with the list of publishers and languages for the form
            return render(request, 'urldb/predict_demand.html', {
                'predicted_demand': predicted_demand,
                'interpreted_demand': demand_label,
                'publishers': self.publishers,
                'languages': self.languages
            })

        except Exception as e:
            # In case of any error, return a JSON response with the error message
            return JsonResponse({'error': str(e)}, status=400)