# LoanPredictionIII - Analytics_Vidhya

Solution to Loan Prediction III Practice Problem on Analytics Vidhya

The problem involves predicting whether a given loan application will be approved or not given information such as the applicant's income, gender, marital status, employment status, loan amount, loan tenure etc.

My approach:

- EDA (described in detail as a ipynb in this repository itself)

1) Looked at individual effect of each variable on the target variable to gain some initial hypothesis of important variables. This included crosstabs for categorical variables and a distribution plot for continuous variables grouped by the target variable value

2) Imputed missing values using group-wise mean/mode and other evidence from pivot tables generated from the data

3) Engineered features such as EMI to Income ratio and total income (applicant + co-applicant income)

4) Used random forests and xgboost linear boosting models on the data. Tuned the parameters using randomized search based on 5-fold cross-validation. Picked the best model out of the two

Results:

Accuracy:

0.79166 (XGBoost)
0.784722 (RF)

Will try genetic algorithm to select the optimal set of features for the model.

