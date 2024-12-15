from django.urls import path
from .views import IndexView, ChurnPredictionView,DemandPredictionView, PricePredictionView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),  # Main page
    path('churn/', ChurnPredictionView.as_view(), name='churn_prediction'),  # Churn Prediction 
    path('predict-price/', PricePredictionView.as_view(), name='predict_price'),
    path('predict-demand/', DemandPredictionView.as_view(), name='predict_demand'),
    ]
