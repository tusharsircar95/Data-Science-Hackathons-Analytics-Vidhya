# Lord Of The Machines
## Data Science Hackathon conducted by Analytics Vidhya

(Contest Link: https://datahack.analyticsvidhya.com/contest/lord-of-the-machines/)

Team: alpha1995

Public LB Score: 0.6837

Private LB Score: 0.6786

Overall Rank: 19/153

Sharing our solution to the Lord Of The Machines data science hackathon conducted by Analytics Vidhya
The problem involved predicting the click probability of links inside a mailer for email campaigns from January 2018 to March 2018.

### Dataset
Details can be found in the contest link mentioned above. We are given details about when a campaign was sent to a particular user, some information about the campaign such as number of links, images, subject, body etc. and whether the user opened and/or clicked the link.
We are supposed to predict the probability of user clicking on a link.

### Data Processing & Exploring
We started by joining the train and test dataset with the campaign data and then engineered the following features:

(1) Hour when email was sent (send_hour)

(2) Day of the week when email was sent (weekday_type)

This was followed by visualizing the probability of opening/clicking of links based on:
(1) Communication Type

 ![CType](https://github.com/tusharsircar95/LordOfTheMachines-Analytics_Vidhya-/blob/master/CType_vs_Prob.png)

(2) Send Hour

![SendHour](https://github.com/tusharsircar95/LordOfTheMachines-Analytics_Vidhya-/blob/master/SendHour_vs_Prob.png)

(3) Day Of The Week

![WeekdayType](https://github.com/tusharsircar95/LordOfTheMachines-Analytics_Vidhya-/blob/master/WeekdayType_vs_Prob.png)


Our first attempt was to calculate the probability that a user will open or click a link based on UserID for each user. For UserID's present in test set that are not present in the training set, I used the mean probability of opening/clicking across all campaigns having the same communication_type. This gave a public leaderboard score of 0.5714.

We then created a basic logistic regression model based on the open and click probabilities(OP and CP) so calculated. This gave a public leaderboard score of 0.6821.

We then generated estimates based on Naive Bayes(NB).
P(Click|Campaign) = P(Campaign|Click) * P(Click) / P(Campaign)
The above were calculated as follows:


Probabilities were calculated and replaced by aggregated proxies when certain values were missing (for example new UserID's in test set).


P(Click): Number of clicks / Number of emails sent to a particular UserID. In case of a new UserID, the mean probability for the email's Communication Type was used as a proxy

P(Campaign): Number of emails for given Campaign_ID / Total number of emails sent (calculate over both train and test)

P(Click|Campaign): This is the most interesting part. We represent a campaign by its features (number of links, images, sections, communication_type etc. Numberical features are convereted to categorical using cut. Let's denote these features as k1,k2 etc.
Therefore, each campaign is [k1, k2, ... kn].

Assuming independence of features,

P([k1, k2, k3 ... kn] | Click) = P([k1] | Click) * P([k2] | Click) ... P([kn] | Click)

P([ki]|Click) is simply the number of times a campaign had feature ki across all campaigns sent to the user. This way we were able to incorporate the features of a campaign into the naive bayes estimate.


The Naive Bayes method gave a public leaderboard score of 0.644.

We finally used the Naive Based estimates as another feature fore the Logistic Regression model and created an ensemble out of the following models:

1) Naive Bayes

2) Logistic Regression (OP , CP , NB)

3) Logistic Regression (OP , CP)

4) Logistic Regression (CP , NB)

A weighted average of the above produced our final output having a public leaderboard score of 0.6837 and a private leaderboard score of 0.6786.

Adding communication type, number of links, images etc. directly as features did not seem to increase our score hence they were not used as features in the logistic regression model. Instead, the NB estimate calculated using these features were incorporated.

### Techniques Not Tried
- Target encoding of categorical variables
- Aggregating text features of email to create a user profile feature
- Features such as time between emails




