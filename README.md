# credit-risk-classification

## Overview of the Analysis

This credit risk classification analysis was created to predict the creditworthiness of borrowers, given their previous history of loans, debt, income, and other financial information. This model was written to predict the liklihood that a borrower was at low-risk (classified by the model using a 0) or high-risk (classified by the model using a 1) for defaulting if granted a new loan.

This machine learning model was constructed by first creating the labels set (`y`)  from the “loan_status” column, and then creating the features (`X`) DataFrame from the remaining columns of the data found in `lending_data.csv`. Then splitting the data into testing and training sets for the model to use. 

The first model tested was a `LogisticRegression` model which was fit by using the training data (`X_train` and `y_train`). Next, the predictions on the testing data labels were saved using the testing feature data (`X_test`) and the fitted model. Finally, the model was evaulated by calculating the accuracy score of the model, generating a confusion matrix, and printing the classification report.

The second model was run in the same manner using `LogisticRegression`, but the `RandomOverSampler` module from the imbalanced-learn library was used to resample the data. This was done to see if the model was more predictive when there were equal numbers of high-risk (1) and low-risk (0) samples in the data set.

## Results

* Machine Learning Model 1 (`LogisticRegression` on original data set):
  * Balanced accuracy score: 0.95
  * Precision score for predicting low-risk borrowers (0):  1.00
  * Precision score for predicting high risk borrowers (1): 0.85
  * Recall score for predicting low-risk borrowers (0): 0.99 
  * Recall score for predicting high-risk borrowers (1): 0.91



* Machine Learning Model 2 (`LogisticRegression` on resampled data set):
  * Balanced accuracy score: 0.99
  * Precision score for predicting low-risk borrowers (0):  1.00
  * Precision score for predicting high risk borrowers (1): 0.84
  * Recall score for predicting low-risk borrowers (0): 0.99 
  * Recall score for predicting high-risk borrowers (1): 0.99

## Summary

Both of these models perform relatively well, with somewhat small differences between them.
