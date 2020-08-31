from django.urls import path, include
from .views import LoanPrediction, retrain_view , process_retrain, process_prediction

app_name = 'loan'

urlpatterns = [
    # path('', admin.site.urls),
    # path('loan/', include('loan.urls')),
    path('predict_loan/', LoanPrediction.as_view(), name='predict_loan'),
    path('retrain/', retrain_view, name='retrain'),
    path('process_retrain/', process_retrain, name='process_retrain'),
    path('process_predict/', process_prediction, name='process_prediction'),
]
