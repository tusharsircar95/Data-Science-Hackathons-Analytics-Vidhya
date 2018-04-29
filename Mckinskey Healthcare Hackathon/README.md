# Mckinskey Healthcare Hackathon by Analytics Vidhya
## Stroke Prediction - Data Science Hackathon

(Contest Link: https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon/)

Sharing my solution to the healthcare hackathon organized by Mckinskey on Analytics Vidhya. We were given several health, demographic and lifestyle details about patients including details such as age and gender, along with several health parameters (e.g. hypertension, body mass index) and lifestyle related variables (e.g. smoking status, occupation type). 
We were supposed to predict the probability of stroke happening to a patients. This would help doctors take proactive health measures for these patients.

Evaluation metric was AUC ROC

Public LB Score: 0.8278

Private LB Score: 0.8577

Overall Rank: 28/584

### Dataset
Details regarding the dataset can be found on the contest site. We were given details of a patient such as smoking status, gender, health conditions, lifestyle conditions etc. and were supposed to predict the probability of suffering a strokg for that patient.

### Pre-processing & EDA
Exploratory data analysis has been presented in a ipython notebook here https://github.com/tusharsircar95/Data-Science-Hackathons-Analytics-Vidhya/blob/master/Mckinskey%20Healthcare%20Hackathon/EDA_MckinskeyHealthcareHackathon_AV.ipynb
I looked at the distribution of all variables, the possible values they can take and how they affect the target variable. Also, went through some studies that indicate which factors are responsible for strokes. Based on that narrowed down a few features that seemed to be important.

Next, I imputed missing values for smoking_status and bmi using a group based mode and median respectively. I created a new feature HTpHD which is the sum of heart_disease and hypertension. The intuition behind this was to create a variable denoting the severeness of a disease affecting the patient.

### Modelling

I have used logistic regression, xgboost trees and catboost models. I tuned the parameters for each using randomized search with 5-old cross-validation. Features were selected based on how they affected the cross-validation accuracy.

To handle the class imbalance I have used class weights inversely proportional to the frequency of that class.

Final features selected were: ['smoking_status, 'age', 'heart_disease', 'hypertension', 'HTpHD', 'avg_glucose_level' ,'gender']

A simple average of the best performing models of each classifier was taken as the final submission.

NOTE: I think the cross-validation strategy helped quite a lot. Eventhough it gave an okayish public leaderboard score, the private leaderboard score went up drastically.

#### Techniques Not Tried
- Thought about using pair-wise features such as Gender+BMI and Gender+Hypertension but didn't implement it.
- Also, categorical variables such (obese, underweight, overweight) or (glucose levels normal, abnormal) could have been used










