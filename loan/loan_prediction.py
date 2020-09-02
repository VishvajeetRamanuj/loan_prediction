#!/usr/bin/env python
# coding: utf-8

# In[1]:


############################################################################
# This program do training on dataset for making decision to give loan or not
# It also do Prediction whether to give loan or not for new record
# Author: Vishvajeet Ramanuj
# Date Created: 26/08/2020
############################################################################


# In[2]:


# importing necessory libreries
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import joblib
import pickle


# In[3]:


# reading data from file
loan_ds = pd.read_csv('loan_ds.csv')


# # spliting data to train and test set

# In[4]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(loan_ds, loan_ds['Loan_Status']):
    strat_train_set = loan_ds.loc[train_index]
    strat_test_set = loan_ds.loc[test_index]


# In[5]:


# function for preprocessing data
def preprocess_loan_ds(dataset):
    if dataset['Loan_Status'].iloc[0]:
#         print('spliting labels')
        dataset['Loan_Status'].iloc[0]
        # spliting labels from data
        label = dataset['Loan_Status'].copy()
        data = dataset.drop('Loan_Status', axis=1)
        label = label == 'Y'
    else:
        print('only data so no need to split label')
        data = dataset

    # preprocessing data
    # filling balnk value with appropriate value and coverting to boolean datatype where only two options
    # first need to fill na values to desired one
    
    # droping index as it is not feature and we are using diffrent index for spliting data
    data = data.drop('Loan_ID', axis=1)

    # converting dependent object to int
    data = data.replace('3+', '3')
    median_dependent = data['Dependents'].median()
    data['Dependents'].fillna(median_dependent, inplace=True)
    data['Dependents'] = data['Dependents'].astype('int')

    # preprocessing Married
    data['Married'].fillna('Yes', inplace=True)

    # converting object to boolean
    data.loc[:,'Married'] = data['Married'] == 'Yes'

    # preprocessing Gender
    data['Gender'].fillna('Male', inplace=True)
    data.loc[:, 'Gender'] = data['Gender'] == 'Male' # setting True for Male and False for Female

    # preprocessing Education
    data.loc[:, 'Education'] = data['Education'] == 'Graduate'

    # preprocessing Self_Employed
    data['Self_Employed'].fillna('No', inplace=True)
    data.loc[:, 'Self_Employed'] = data['Self_Employed'] == 'Yes'

    # preprocessing Credit_History
    data['Credit_History'].fillna(1, inplace=True)
    data.loc[:, 'Credit_History'] = data['Credit_History'] == 1

    # preprocessing Loan_Amount_Term
    data['Loan_Amount_Term'].fillna(360, inplace=True)
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].astype('int')

    # preprocessing LoanAmount
    data = data.replace('3+', '3')
    median_dependent = data['Dependents'].median()
    data['Dependents'].fillna(median_dependent, inplace=True)
    data['Dependents'] = data['Dependents'].astype('int')
    
    loan_amount_mean = data['LoanAmount'].mean()
    data['LoanAmount'] = data['LoanAmount'].fillna(loan_amount_mean)
    data['LoanAmount'] = data['LoanAmount'].astype('int')

    # preprocessing Property_Area
    strat_train_cat = data[['Property_Area']]

    cat_encoder = OneHotEncoder(sparse=False)
    strat_train_cat_encoded = cat_encoder.fit_transform(strat_train_cat)
    
    # saving encoder
    with open('encoder.txt', 'wb') as f:
        pickle.dump(cat_encoder, f)

    strat_train_cat_df = pd.DataFrame(strat_train_cat_encoded, index=data.index, columns=['Rural', 'Semiurban', 'Urban'])

    frames = [data, strat_train_cat_df]
    new_ds = pd.concat(frames, axis=1)

    new_ds.drop('Property_Area', axis=1, inplace=True)
    
    if dataset['Loan_Status'].iloc[0]:
        return (new_ds, label)
    else:
        return new_ds

    


# In[6]:


# train_data
# train_data = train_data.drop('Loan_ID', axis=1)


# In[7]:


train_data, train_labels = preprocess_loan_ds(strat_train_set)


# In[8]:


# verifying proportation of stratified training set is the same as original dataset
print('train set')
strat_train_set['Loan_Status'].value_counts() / len(strat_train_set)

print('original')
loan_ds['Loan_Status'].value_counts() / len(loan_ds)


# # Selecting Model and Training

# In[9]:


# training Random forest Regressor model
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(train_data, train_labels)


# In[10]:


# saving traied model
joblib.dump(forest_reg, 'forest_reg.pkl')


# In[11]:


test_data, test_labels = preprocess_loan_ds(strat_test_set)


# In[14]:


predictions = forest_reg.predict(test_data)


# In[15]:


# predictions
score = accuracy_score(test_labels, predictions.round(), normalize=False)
score # there must be something wrong


# In[16]:


# loading encoder
file = open('encoder.txt', 'rb')
cat_encoder = pickle.load(file)

# loading model
forest_reg = joblib.load("forest_reg.pkl")


# In[17]:


# demo new label prediction
a = [['Urban']]
type(a)
# print(a)
cat_encoder.transform(a)


# In[18]:


# Single Prediction
ID = "LP002991"
Gender = "Male"
Married = "Yes"
Dependents = 2
Education = "Graduate"
Self_Employed = "Yes"
ApplicantIncome = 5000
CoapplicantIncome = 2000
LoanAmount = 250
Loan_Amount_Term = 360
Credit_History = 1
Property_Area = "Urban"
# Loan_Status = ?

def loan_grant_decision(ID, ApplicantIncome, CoapplicantIncome, Property_Area, Gender="Male", Married="Yes", 
                        Dependents=0, Education = "Graduate", Self_Employed='No',
                        LoanAmount=147, Loan_Amount_Term=360, Credit_History='Yes', threshhold=0.70
                        ):
    # spliting labels from data
    loan_data = pd.DataFrame([[
                              Gender, Married, Dependents, Education, Self_Employed, 
                              ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, 
                              Credit_History, Property_Area
                             ]], 
                             columns = [
                              'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                              'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                              'Loan_Amount_Term', 'Credit_History', 'Property_Area'
                             ])

    # converting object to boolean
    loan_data.loc[:, 'Married'] = loan_data['Married'] == 'Yes'

    # preprocessing Gender
    loan_data.loc[:, 'Gender'] = loan_data['Gender'] == 'Male'

    # preprocessing Education
    loan_data.loc[:, 'Education'] = loan_data['Education'] == 'Graduate'

    # preprocessing Self_Employed
    loan_data.loc[:, 'Self_Employed'] = loan_data['Self_Employed'] == 'Yes'

    # preprocessing Credit_History
    loan_data.loc[:, 'Credit_History'] = loan_data['Credit_History'] == 1
    loan_data_cat = loan_data['Property_Area'].to_numpy().reshape(-1,1)

    loan_data_cat_encoded = cat_encoder.transform(loan_data_cat)
    loan_data_cat_array = loan_data_cat_encoded
    loan_data_cat_df = pd.DataFrame(loan_data_cat_array, index=loan_data.index, columns=['Rural', 'Semiurban', 'Urban'])

    frames = [loan_data, loan_data_cat_df]
    new_ds = pd.concat(frames, axis=1)

    new_ds.drop('Property_Area', axis=1, inplace=True)
#     new_ds = preprocess_loan_ds(loan_ds) # you can replace above preprocessing with this function
    prediction = forest_reg.predict(new_ds)
    if prediction[0] >= threshhold:
        return True # grant loan
    else:
        return False # do not grant loan


# In[19]:


result = loan_grant_decision(ID, ApplicantIncome, CoapplicantIncome, Property_Area, Gender=Gender, Married=Married, 
                             Dependents=Dependents, Education = Education,
                             Self_Employed=Self_Employed, LoanAmount=LoanAmount,
                             Loan_Amount_Term=Loan_Amount_Term, Credit_History=Credit_History)


# In[20]:


# result


# In[21]:


# Retrain
def retrain(new_ds_file):
    # reading new_csv
    new_dataset = pd.read_csv(new_ds_file)

    # droping index as it is not feature and we are using diffrent index for spliting data
    new_dataset = new_dataset.drop('Loan_ID', axis=1)
    
    old_dataset = pd.read_csv('loan_ds.csv')

    combine_ds = pd.concat([old_dataset, new_dataset])
    # store combine_ds

    # do preprocessing
    combine_ds_data, combine_ds_labels = preprocess_loan_ds(combine_ds)

    # we are creating new model which will be trained on all data
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42) 
    forest_reg.fit(train_data, train_labels)

    # saving new model
    joblib.dump(forest_reg, 'forest_reg.pkl')
    return True


# In[22]:


retrain('loan_ds_2.csv')

