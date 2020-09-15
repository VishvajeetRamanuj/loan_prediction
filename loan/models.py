from django.db import models

RURAL = 'Rural'
SEMIURBAN = 'Semiurban'
URBAN = 'Urban'
MALE = 'Male'
FEMALE = 'Female'
GRADUATE = 'Graduate'
NOT_GRADUATE = 'Not Graduate'   

PROPERTY_AREA_CHOICE = (
    (RURAL, 'Rural'),
    (SEMIURBAN, 'Semiurban'),
    (URBAN, 'Urban'),
)

GENDER_CHOICE = (
    (MALE, 'Male'),
    (FEMALE, 'Female'),
)

EDUCATION_CHOICE = (
    (GRADUATE, 'Graduate'),
    (NOT_GRADUATE, 'Not Graduate')
)

# Create your models here.
class LoanPredictionModel(models.Model):
    loan_id = models.CharField(max_length=8, blank=False)
    gender = models.CharField(max_length=6, choices=GENDER_CHOICE, blank=False)
    married = models.BooleanField(blank=False)
    dependents = models.IntegerField(default=0)
    education = models.CharField(max_length=13, choices=EDUCATION_CHOICE, blank=False)
    self_employed = models.BooleanField(blank=False)
    applicant_income = models.IntegerField(default=0, blank=False)
    coapplicant_income = models.IntegerField(default=0, blank=False)
    loan_amount = models.IntegerField(default=0, blank=False)
    loan_amount_term = models.IntegerField(default=0, blank=False) #days
    credit_history = models.BooleanField(default=0, blank=False)
    property_area = models.CharField(max_length=9, choices=PROPERTY_AREA_CHOICE, blank=False)