from django import forms
from .models import LoanPredictionModel

class LoanPredictionForm(forms.ModelForm):
    # def __init__(self, *args, **kwargs):
    #     super(LoanPredictionForm, self).__init__(*args, **kwargs)
    #     for i in ('loan_id','gender','married','dependents','education','self_employed','applicant_income','coapplicant_income','loan_amount','loan_amount_term','credit_history','property_area'):
    #         self.fields[i].required = False
    
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
        # help_texts = {
        #     'loan_id':None,
        #     'gender':None,
        #     'married':None,
        #     'dependents':None,
        #     'education':None,
        #     'self_employed':None,
        #     'applicant_income':None,
        #     'coapplicant_income':None,
        #     'loan_amount':None,
        #     'loan_amount_term':None,
        #     'credit_history':None,
        #     'property_area':None
        # }

    