import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.columns)

print('Missing values per attribute in train:')
print(train.apply(lambda x: x.isnull().sum(), axis=0) * 100 / len(train))

print('Rows with missing values in train:')
print(train.apply(lambda x: (x.isnull().sum() > 0), axis=1).sum() * 100 / len(train))


print('Missing values per attribute in test:')
print(test.apply(lambda x: x.isnull().sum(), axis=0) * 100 / len(test))

print('Rows with missing values in test:')
print(test.apply(lambda x: (x.isnull().sum() > 0), axis=1).sum() * 100 / len(train))

# Let's how each variable affects the target variable Loan_Status' individually
# Map Y to 1 and N to 0
train.Loan_Status = train.apply(lambda x: (1 if x['Loan_Status'] == 'Y' else 0),axis=1)
# See percentage of loans apporved
print(train.Loan_Status.mean() * 100)

# Continuous
field = 'ApplicantIncome'
sns.distplot(train[train.Loan_Status == 1][field],color='g')
sns.distplot(train[train.Loan_Status == 0][field],color='r')
plt.legend(['Approve','Rejected'])
plt.show()

field = 'CoapplicantIncome'
sns.distplot(train[train.Loan_Status == 1][field],color='g')
sns.distplot(train[train.Loan_Status == 0][field],color='r')
plt.legend(['Approve','Rejected'])
plt.show()

field = 'LoanAmount'
sns.distplot(train[(~train.LoanAmount.isnull()) & (train.Loan_Status == 1)][field],color='g')
sns.distplot(train[(~train.LoanAmount.isnull()) & (train.Loan_Status == 0)][field],color='r')
plt.legend(['Approve','Rejected'])
plt.show()

field = 'Loan_Amount_Term'
sns.distplot(train[(~train.Loan_Amount_Term.isnull()) & (train.Loan_Status == 1)][field],color='g')
sns.distplot(train[(~train.Loan_Amount_Term.isnull()) & (train.Loan_Status == 0)][field],color='r')
plt.legend(['Approve','Rejected'])
plt.show()

print( train.pivot_table(index='Gender',values='Loan_Status',aggfunc=('mean','count')) )

print( train.pivot_table(index='Married',values='Loan_Status',aggfunc=('mean','count')) )

print( train.pivot_table(index='Dependents',values='Loan_Status',aggfunc=('mean','count')) )

print( train.pivot_table(index='Education',values='Loan_Status',aggfunc=('mean','count')) )

print( train.pivot_table(index='Self_Employed',values='Loan_Status',aggfunc=('mean','count')) )

print( train.pivot_table(index='Credit_History',values='Loan_Status',aggfunc=('mean','count')) )

print( train.pivot_table(index='Property_Area',values='Loan_Status',aggfunc=('mean','count')) )

print( train.pivot_table(index='Loan_Amount_Term',values='Loan_Status',aggfunc=('mean','count')) )

# On first observation:
# Credit_History, Education, Married, Gender, Dependents and Property_Area seem to affect the chance of getting a loan approved
# Credit_History seems to be the most significant variable and also has the most % of missing values which need to be imputed

# Imputing Missing Values
# Gender
# Marreid
# Dependents
# Self_Employed
# LoanAmount
# Loan_Amount_Term
# Credit_History

# Credit_History seems to be the most important variable. 0 -> 0 and 1-> 1 hence we can use that to impute missing values in the train set
train.loc[(train.Credit_History.isnull()) & (train.Loan_Status == 1),'Credit_History'] = 1
train.loc[(train.Credit_History.isnull()) & (train.Loan_Status == 0),'Credit_History'] = 0
# This takes care of missing values in the train set
# Now let's look at the combined train and test set to impute test set
combined = pd.concat([train,test])

# Let's see if credit_history is determined by any other variable
cols = ['Self_Employed', 'Gender', 'Married' ,'Education', 'Dependents', 'Property_Area']
pivots = []
for col in cols:
    pivots.append( ( combined.pivot_table(index=col,values='Credit_History',aggfunc=('mean')) ) )

for pivot in pivots:
    pivot.plot(kind='bar')
    plt.show()

field = 'ApplicantIncome'
sns.distplot(combined[(~combined.ApplicantIncome.isnull()) & (combined.Credit_History == 1)][field],color='g')
sns.distplot(combined[(~combined.ApplicantIncome.isnull()) & (combined.Credit_History == 0)][field],color='r')
plt.legend(['Credit_History 1','Credit_History 2'])
plt.show()

# Can't see any clear distinction
# Let's replace missing values with groupwise mode
pivot = train.pivot_table(
        index=['Self_Employed', 'Property_Area', 'Dependents', 'Gender', 'Education', 'Married'],
        values='Credit_History', aggfunc=('mean', 'count'))

missed = 0
chPredictions = []
for i in range(len(test)):
    if np.isnan(test.loc[i,].Credit_History):#.isnull():
        #print(i)
        chance = 0.0
        try:
            chance = pivot.loc[test.loc[i,].Self_Employed].loc[test.loc[i,].Property_Area].loc[test.loc[i,].Dependents].loc[test.loc[i,].Gender].loc[test.loc[i,].Education].loc[test.loc[i,].Married]['mean']
        except:
            chance = 1.0
            missed = missed + 1
        pred = 1.0 if chance > 0.75 else 0.0
        chPredictions.append(pred)
        test.loc[i,].Credit_History = pred
#print(missed)

test.loc[np.isnan(test.Credit_History),"Credit_History"] = chPredictions

# Let's look at other pair-plots
# Let's look at other missing variables
missingColsCat = ['Gender', 'Married', 'Dependents', 'Self_Employed','Credit_History']
for i in range(len(missingColsCat)):
    for j in range(len(missingColsCat)):
        if i != j:
            pd.crosstab(train[missingColsCat[i]],train[missingColsCat[j]]).plot(kind='bar')
            plt.show()
# We will look for variables such that the distribution of the other variable grouped by this variable is different.
# # Ow it means that the first variable doesn't affect the second variable
combined = pd.concat([train,test])
# Given gender we can talk about marriage distinctly. Male -> Married and Female -> Not Married
combined.loc[(combined.Gender.isnull()) & (combined.Gender == 'Male'),'Married'] = 'Yes'
combined.loc[(combined.Gender.isnull()) & (combined.Gender == 'Female'),'Married'] = 'No'
# Let's impute other missung values with the mode
combined = pd.concat([train,test])
combined.Gender.fillna('Male',inplace=True)
combined.Married.fillna('Yes',inplace=True)
combined.Dependents.fillna('0',inplace=True)
combined.Self_Employed.fillna('No',inplace=True)
combined.LoanAmount.fillna(np.mean(combined.LoanAmount),inplace=True)
combined.Loan_Amount_Term.fillna(360.0,inplace=True)

train,test = combined[:len(train)],combined[len(train):]
test = test.drop(['Loan_Status'],axis=1)

# Check if missing values have been removed
print('Rows with missing values in train:')
print(train.apply(lambda x: (x.isnull().sum() > 0), axis=1).sum() * 100 / len(train))
print('Rows with missing values in test:')
print(test.apply(lambda x: (x.isnull().sum() > 0), axis=1).sum() * 100 / len(test))

# Let's try to engineer some features
# Loan_Amount and Application/Coapplicant Income don't seem to make a difference which is surprising. Maybe the ratio of Loan_Amount to the income might be important
train.loc[:,'TotalIncome'] = train.ApplicantIncome + train.CoapplicantIncome
train.loc[:,'ILR'] = train.TotalIncome / train.LoanAmount

field = 'ILR'
sns.distplot(train[(~train.ILR.isnull()) & (train.Loan_Status == 1)][field],color='g')
sns.distplot(train[(~train.ILR.isnull()) & (train.Loan_Status == 0)][field],color='r')
plt.legend(['Approve','Rejected'])
plt.show()

# Even this doesn't seem to have an effect!
# Since the loan is paid over several years in monthly installments, the ratio of the income to the EMI might be more significant.
# Also, rate of interest for the loan should also be considered, let's assume it to be 7%
interestRate = 0.07
# Get the number of years for which the loan is sanctioned
train.Loan_Amount_Term = train.Loan_Amount_Term / 12
# Even this doesn't seem to have an effect!
# Since the loan is paid over several years in monthly installments, the ratio of the income to the EMI might be more significant.
# Also, rate of interest for the loan should also be considered, let's assume it to be 7%
interestRate = 0.07
# Get the number of years for which the loan is sanctioned
train.Loan_Amount_Term = train.Loan_Amount_Term / 12
train.loc[:,'LoanRepayAmount'] = train.apply(lambda x: x['LoanAmount']*((1 + interestRate)**x['Loan_Amount_Term']),axis=1)
train.loc[:,'LoanEMI'] = train.apply(lambda x: x['LoanAmount']*interestRate*((1 + interestRate)**x['Loan_Amount_Term']/((1+interestRate)**(x['Loan_Amount_Term']-1))),axis=1)
train.loc[:,'IEMIR'] = train.TotalIncome / train.LoanEMI

field = 'IEMIR'
sns.distplot(train[(~train.IEMIR.isnull()) & (train.Loan_Status == 1)][field],color='g')
sns.distplot(train[(~train.IEMIR.isnull()) & (train.Loan_Status == 0)][field],color='r')
plt.legend(['Approve','Rejected'])
plt.show()

# Still doesn't seem to be very significant, but intuitively this IEMIR variable should matter. We'll explore it further later.
