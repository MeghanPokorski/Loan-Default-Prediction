# Loan-Default-Prediction

Predicting loan defaults helps lenders make more informed decisions with current and future borrowers. The lender can implement mitigation strategies if a current borrower moves into a higher risk category. For example, job loss or a decrease in income would make a current borrower more likely to default on their loan.

The lender can also use the default prediction model to decide whether or not a given loan amount and term is too high risk for a loan applicant.

## Purpose

The purpose of this project was to create a binary classification model to predict loan defaults.

## Methods Used

- Exploratory Data Analysis
- Predictive Modeling

## Tools

- python
    - Pandas
    - NumPy
    - Matplotlib
    - Scikit-learn
    - Seaborn 

## Data

Since real customer banking information is not available to the public, the models will be trained using a dataset originally created by [Coursera](https://www.coursera.org/projects/data-science-coding-challenge-loan-default-prediction?action=enroll) The dataset is also available on [Kaggle](https://www.kaggle.com/datasets/nikhille9/loan-default/data)

## Results Summary

Of the five models tested, Support Vector Machine produced the best results. It had the highest accuracy, but low precision, recall, ROC AUC, and F1:
<br>
<br>
Test Accuracy: 0.8075
<br>
Test Precision: 0.1559633027522936
<br>
Test Recall: 0.10897435897435898
<br>
Test ROC AUC: 0.5104258768051871
<br>
Test F1: 0.12830188679245283
<br>
<br>
The SVM model predicted 952 of the 1066 negatives in the test dataset, however it only got 17 of 134 positives.


## Inaccuarcy caused by SMOTE

SMOTE, or Synthetic Minority Over-sampling, is widely recommended for unbalanced datasets where the one class makes up a disproportinatly large percentage of the training data. SMOTE creates synthetic values for the minority class so the training set is a 50/50 split. The idea is that it will create more information with which the model can be trained. This is the case for older, "weaker" models, like SVM where the calibration of the training data isn't a factor in the estimate. However, for more modern "strong' models like XGBoost and Random Forest the calibration of the training set is important to accurately predicting the results. This impact can be seen in the results of these models where they predicted disproportinatly in favor of default.

## Conclusions and Recommendations

While SVM produced the most accurate results for this project, for real world application it may not be the best option. This project limited the training data to a small subset of the available 250k data points to improve speed, but this is not a realistic strategy when training a production model. Since SVM does produce better results with SMOTE, a 250k entry data set with a 90/10 split would become a 450k entry training set (225k of the majority class, 25k real entries for the minority class, and 200k synthetic entries for the minority class.) Taking into consideration the actual [delinquency rate](https://fred.stlouisfed.org/series/DRALACBN) for loans in the US, which is currently around 1.5% and peaked at 7.5% following the 2008 housing crisis, the number of synthetic data points required to balance the training data SVM quickly becomes too computationally intensive to be useful in this use case.

For a production loan default predictor, we would recommend using one of the "stronger" models like XGBoost, Random Forest, or even a [deep learning model](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0192-5). However instead of applying SMOTE, the imbalanced classes can be addressed with cost-sensitive learning or weighting the classes in model tuning.
