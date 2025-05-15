# Bike Classification || 1RT700
Given weather parameters like temperature, wind speed, dew, etc., plot trends and predict **'demand for bike usage'**.    
Add SMOTE and Feature expansion to reduce model bias and better fit the training dataset. Report precision, recall,    
and f1-score to compare model performance. Regression model with ```feature expansion``` was observed to perform   
better than models like vanilla regression and ```SMOTE-regression```. To replicate the results, clone the repository, then   
execute the `model.py` file.

---
***Observations***   
> Monday - Friday (Weekdays)
* High demand for bikes between 6 AM and 9 PM.
* Demand peaks between 5 PM AND 7 PM.
* Demand flatlines < 6 AM and > 9 PM.

> Saturday and Sunday (Weekends)
* Consistently high demand from 8 AM to 8 PM.

> Holidays
* High demand for bikes between 9 AM and 5 PM.
* Peaks at around 12 PM and 3 PM.

> Correlation
* Temperature and dew are highly positively correlated.

---
***Results***
| Model | F1-score on validation set |
| :- | :- |
| Logistic Regression | 0.85 |
| Regression with SMOTE | 0.82 |
| Regression with feature expansion | 0.86 |
