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
    # form = GroupForm(use_required_attribute=False)
    form_class = LoanPredictionForm # (request.POST or None) # use_required_attribute=False
    template_name = 'loan/loan_decision_prediction.html'

def process_prediction(request, *args, **kwargs):
    if request.method == "POST":
    # form = LoanPredictionForm() #(request.POST or None)
    # if form.is_valid():
        # form.save()
        loan_id = request.POST.get("loan_id","")
        gender = request.POST.get("gender","")
        married = request.POST.get("married", "No")
        dependents = int(request.POST.get("dependents",""))
        education = request.POST.get("education","")
        self_employed = request.POST.get("self_employed", "No")
        applicant_income = int(request.POST.get("applicant_income",""))
        coapplicant_income = int(request.POST.get("coapplicant_income",""))
        loan_amount = int(request.POST.get("loan_amount",""))
        loan_amount_term = int(request.POST.get("loan_amount_term",""))
        credit_history = int(request.POST.get("credit_history",1))
        property_area = request.POST.get("property_area","")

        print(loan_id, applicant_income,coapplicant_income,property_area, gender, married,
                                        dependents, education, self_employed, loan_amount, loan_amount_term, credit_history)

        print(type(loan_id), type(applicant_income), type(coapplicant_income), type(property_area), type(gender), type(married),
                                        type(dependents), type(education), type(self_employed), type(loan_amount), type(loan_amount_term), type(credit_history))

        result = loan_grant_decision(loan_id, applicant_income,coapplicant_income,property_area, gender, married,
                                        dependents, education, self_employed, loan_amount, loan_amount_term, credit_history)
        # result = loan_grant_decision('LP001002', 5849, 0, 'Urban', 'Male', 'No', 0, 'Graduate', 'No', 100, 360, 1)
                                        
        # result = True
    # def loan_grant_decision(ID, ApplicantIncome, CoapplicantIncome, Property_Area, Gender="Male", Married="Yes", 
    #                     Dependents=0, Education = "Graduate", Self_Employed='No',
    #                     LoanAmount=147, Loan_Amount_Term=360, Credit_History='Yes', threshhold=0.70
    #                     ):

        # result = loan_grant_decision('LP001002', 5849, 0, 'Urban', 'Male', 'No', 0, 'Graduate','No', 100, 360, 1)
                                        # LP001002 5849 0 Urban Male No 0 Graduate No 100 360 0
        print(result) # hear is bug it is always return False the same method from another program return proper result
        
        if result:
            return HttpResponse("<h1 align=center>give loan</h1>")
        else:
            return HttpResponse("<h1 align=center>do not give loan</h1>")
    else:
        return HttpResponse("<h1 align=center>Try again with post</h1>")

def retrain_view(request, *args, **kwargs):
    return render(request, 'loan/retrain.html')

def process_retrain(request):
    if request.method == "POST":
        new_csv = request.FILES['new_csv'].name
        # retraining
        print(new_csv)
        result = retrain(new_csv) # True
        if result:
            return HttpResponse("<h1 align=center>Training is done successfully</h1>")
        else:
            return HttpResponse("<h1 align=center>Try again uploading file</h1>")
    else:
        return HttpResponse("<h1 align=center>Try uploading file using submit</h1>") # render to home