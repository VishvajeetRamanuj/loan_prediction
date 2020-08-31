from django.shortcuts import render
from django.views.generic.edit import CreateView
from django.http import HttpResponse
from .models import LoanPredictionModel
from .forms import LoanPredictionForm

from .loan_prediction import loan_grant_decision, retrain

# Create your views here.
def home(request):
    return render(request, 'loan/home.html')

class LoanPrediction(CreateView):
    model = LoanPredictionModel
    form_class = LoanPredictionForm
    template_name = 'loan/loan_decision_prediction.html'

def process_prediction(request, *args, **kwargs):
    if request.method == "POST":
            loan_id = request.POST.get("loan_id","")
            gender = request.POST.get("gender","")
            married = request.POST.get("married","")
            dependents = request.POST.get("dependents","")
            education = request.POST.get("education","")
            self_employed = request.POST.get("self_employed","")
            applicant_income = request.POST.get("applicant_income","")
            coapplicant_income = request.POST.get("coapplicant_income","")
            loan_amount = request.POST.get("loan_amount","")
            loan_amount_term = request.POST.get("loan_amount_term","")
            credit_history = request.POST.get("credit_history","")
            property_area = request.POST.get("property_area","")

            result = loan_grant_decision(loan_id, applicant_income,coapplicant_income,property_area, gender, married,
                                         dependents, education, self_employed, loan_amount, loan_amount_term, credit_history)
            # print(result) # hear is bug it is always return False the same method from another program return proper result
            
            if result:
                return HttpResponse("<h1>give loan</h1>")
            else:
                return HttpResponse("<h1>do not give loan</h1>")
    else:
        return HttpResponse("<h1>Try again with post</h1>")

def retrain_view(request, *args, **kwargs):
    return render(request, 'loan/retrain.html')

def process_retrain(request):
    if request.method == "POST":
        new_csv = request.FILES['new_csv'].name
        # retraining
        print(new_csv)
        result = retrain(new_csv) # True
        if result:
            return HttpResponse("<h1>Training is done successfully</h1>")
        else:
            return HttpResponse("<h1>Try again uploading file</h1>")
    else:
        return HttpResponse("<h1>Try uploading file using submit</h1>") # render to home