from django.urls import path, include
from .views import LoanPrediction, retrain, process_retrain

app_name = 'loan'

urlpatterns = [
    # path('', admin.site.urls),
    # path('loan/', include('loan.urls')),
    path('predict_loan/', LoanPrediction.as_view(), name='predict_loan'),
    path('retrain/', retrain, name='retrain'),
    path('process_retrain', process_retrain, name='process_retrain'),
]
