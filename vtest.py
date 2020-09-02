from loan.loan_prediction import retrain, loan_grant_decision

# result = retrain('loan_ds_2.csv')

loan_id = "LP002991"
gender = "Male"
married = "Yes"
dependents = 2
education = "Graduate"
self_employed = "Yes"
applicant_income = 5000
coapplicant_income = 2000
loan_amount = 250
loan_amount_term = 360
credit_history = 1
property_area = "Urban"

result = loan_grant_decision(loan_id, applicant_income,coapplicant_income,property_area, gender, married,
                                         dependents, education, self_employed, loan_amount, loan_amount_term, credit_history)
print(result)