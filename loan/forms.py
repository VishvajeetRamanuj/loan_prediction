from django import forms
from .models import LoanPredictionModel

class LoanPredictionForm(forms.ModelForm):
    
    class Meta:
        model = LoanPredictionModel
        fields = (
            'loan_id',
            'gender',
            'married',
            'dependents',
            'education',
            'self_employed',
            'applicant_income',
            'coapplicant_income',
            'loan_amount',
            'loan_amount_term',
            'credit_history',
            'property_area'
        )

    