#!/usr/bin/env python
# coding: utf-8

# In[2]:


############################################################################
# This program do training on dataset for making decision to give loan or not
# It also do Prediction whether to give loan or not for new record
# Author: Vishvajeet Ramanuj
# Date Created: 26/08/2020
############################################################################


# In[3]:


# importing necessory libreries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import joblib
import pickle


# In[4]:


# reading data from file
loan_ds = pd.read_csv('loan_ds.csv')


# In[5]:


# droping index as it is not feature and we are using diffrent index for spliting data
loan_ds = loan_ds.drop('Loan_ID', axis=1)


# # spliting data to train and test set

# In[6]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(loan_ds, loan_ds['Loan_Status']):
    strat_train_set = loan_ds.loc[train_index]
    strat_test_set = loan_ds.loc[test_index]


# In[7]:


# function for preprocessing data
def preprocess_loan_ds(dataset):
    if strat_test_set['Loan_Status'].iloc[0]:
        print('spliting labels')
        strat_test_set['Loan_Status'].iloc[0]
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
#     loan_amt_imputer = SimpleImputer(strategy='mean')
#     strat_train_loan_amt = data['LoanAmount'].copy()
#     strat_train_loan_amt_1 = strat_train_loan_amt.to_numpy().reshape(-1,1)
#     loan_amt_imputer.fit(strat_train_loan_amt_1)

#     strat_train_loan_amt_trans = loan_amt_imputer.transform(strat_train_loan_amt_1)

    data = data.replace('3+', '3')
    median_dependent = data['Dependents'].median()
    data['Dependents'].fillna(median_dependent, inplace=True)
    data['Dependents'] = data['Dependents'].astype('int')
    
#     strat_train_loan_amt.mean()
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

#     data.drop('LoanAmount', axis=1, inplace=True)

    # combine data
#     loan_amt_df = pd.DataFrame(strat_train_loan_amt_trans, index=data.index, columns=['LoanAmount'])
#     strat_train_cat_array = strat_train_cat_encoded
    strat_train_cat_df = pd.DataFrame(strat_train_cat_encoded, index=data.index, columns=['Rural', 'Semiurban', 'Urban'])

    frames = [data, strat_train_cat_df]
    new_ds = pd.concat(frames, axis=1)

    new_ds.drop('Property_Area', axis=1, inplace=True)
    
    if strat_test_set['Loan_Status'].iloc[0]:
        return (new_ds, label)
    else:
        return new_ds

    


# In[8]:


train_data, train_labels = preprocess_loan_ds(strat_train_set)


# In[9]:


# verifying proportation of stratified training set is the same as original dataset
print('train set')
strat_train_set['Loan_Status'].value_counts() / len(strat_train_set)

print('original')
loan_ds['Loan_Status'].value_counts() / len(loan_ds)


# # Selecting Model and Training

# In[10]:


# training Random forest Regressor model
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(train_data, train_labels)


# In[11]:


# saving traied model
joblib.dump(forest_reg, 'forest_reg.pkl')


# In[12]:


test_data, test_labels = preprocess_loan_ds(strat_test_set)


# In[13]:


# if strat_test_set['Loan_Status'].iloc[0]:
#     print('yes')
# strat_test_set['Loan_Status'].iloc[0]


# In[14]:


# # preprocessing test data
# strat_test_label = strat_test_set['Loan_Status'].copy()
# strat_test_data = strat_test_set.drop('Loan_Status', axis=1)

# # converting dependent object to int
# strat_test_data = strat_test_data.replace('3+', '3')
# strat_test_data['Dependents'].fillna(median_dependent, inplace=True)
# strat_test_data['Dependents'] = strat_test_data['Dependents'].astype('int')

# # preprocessing Married
# strat_test_data['Married'].fillna('Yes', inplace=True)

# # converting object to boolean
# strat_test_data.loc[:,'Married'] = strat_test_data['Married'] == 'Yes'

# # preprocessing Gender
# strat_test_data['Gender'].fillna('Male', inplace=True)
# strat_test_data.loc[:, 'Gender'] = strat_test_data['Gender'] == 'Male'

# # preprocessing Education
# strat_test_data.loc[:, 'Education'] = strat_test_data['Education'] == 'Graduate'

# # preprocessing Self_Employed
# strat_test_data['Self_Employed'].fillna('No', inplace=True)
# strat_test_data.loc[:, 'Self_Employed'] = strat_test_data['Self_Employed'] == 'Yes'

# # preprocessing Credit_History
# strat_test_data['Credit_History'].fillna(1, inplace=True)
# strat_test_data.loc[:, 'Credit_History'] = strat_test_data['Credit_History'] == 1

# # preprocessing Loan_Amount_Term
# strat_test_data['Loan_Amount_Term'].fillna(360, inplace=True)
# strat_test_data['Loan_Amount_Term'] = strat_test_data['Loan_Amount_Term'].astype('int')

# mean = 147.30997877
# # LoanAmout
# strat_test_data['LoanAmount'].fillna(mean, inplace=True)

# # preprocessing Property_Area
# strat_test_cat = strat_test_data[['Property_Area']]
# strat_test_cat_encoded = cat_encoder.transform(strat_test_cat)

# # combining encoded category with dataset
# strat_test_cat_array = strat_test_cat_encoded
# strat_test_cat_df = pd.DataFrame(strat_test_cat_array, index=strat_test_data.index, columns=['Rural', 'Semiurban', 'Urban'])

# frames = [strat_test_data, strat_test_cat_df]
# new_ds_test = pd.concat(frames, axis=1)

# new_ds_test.drop('Property_Area', axis=1, inplace=True)

# # new_ds

# strat_test_label = strat_test_label == 'Y'


# In[15]:


predictions = forest_reg.predict(test_data)


# In[16]:


# predictions
score = accuracy_score(test_labels, predictions.round(), normalize=False)
score # there must be something wrong


# In[18]:


# demo new label prediction
a = [['Urban']]
type(a)
# print(a)
# cat_encoder.transform(a)


# In[21]:


# loading encoder
file = open('encoder.txt', 'rb')
cat_encoder = pickle.load(file)

# loading model
forest_reg = joblib.load("forest_reg.pkl")


# In[22]:


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


# In[23]:


result = loan_grant_decision(ID, ApplicantIncome, CoapplicantIncome, Property_Area, Gender=Gender, Married=Married, 
                             Dependents=Dependents, Education = Education,
                             Self_Employed=Self_Employed, LoanAmount=LoanAmount,
                             Loan_Amount_Term=Loan_Amount_Term, Credit_History=Credit_History)


# In[24]:


result


# In[25]:


# Retrain



# reading new_csv
new_dataset = pd.read_csv('loan_ds_2.csv')

# droping index as it is not feature and we are using diffrent index for spliting data
new_dataset = new_dataset.drop('Loan_ID', axis=1)

# spliting data into training and test set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(loan_ds, loan_ds['Loan_Status']):
    strat_train_set = loan_ds.loc[train_index]
    strat_test_set = loan_ds.loc[test_index]

# pre process data    
train_data, train_labels = preprocess_loan_ds(strat_train_set)



# In[ ]:


# store previous pre process data in csv
# load previous pre_process data
# combine both dataframe
# do preprocessing
# do training
# save new preprcess data, and model for future reference


# In[28]:


old_dataset = pd.read_csv('loan_ds.csv')

combine_ds = pd.concat([old_dataset, new_dataset])

# do preprocessing
combine_ds_data, combine_ds_labels = preprocess_loan_ds(combine_ds)

# we are creating new model which will be trained on all data
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42) 
forest_reg.fit(train_data, train_labels)

# saving new model
joblib.dump(forest_reg, 'forest_reg.pkl')


# In[ ]:




