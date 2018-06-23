# Home-Credit
22 June 2018

Home Credit is a non-banking financial institution, founded in 1997 in the Czech Republic.

The company operates in 14 countries (including United States, Russia, Kazahstan, Belarus, China, India) and focuses on lending primarily to people with little or no credit history which will either not obtain loans or became victims of untrustworthly lenders.

Home Credit group has over 29 million customers, total assests of 21 billions Euro, over 160 millions loans.

The company uses of a variety of alternative data - including telco and transactional information - to predict their clients' repayment abilities.

The goal of this competition is to help them unlock the full potential of their data. 

## Submission
This afternoon, every team need to submit their prediction on the Kaggle and get a feedback of the Leaderboard rank and AUC score. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target. For each SK_ID_CURR in the test set, you must predict a probability for the TARGET variable. The file should contain a header and have the following format: 
```
SK_ID_CURR,TARGET
100001,0.1
100005,0.9
100013,0.2
etc.
```

## Tasks

- EDA
- Logistic model
- ROC and Area Under the Curve
- Cross validation
- Feature engineering
