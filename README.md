# Bike Classification || 1RT700
Given weather parameters like temperature, wind speed, dew, etc., plot trends and predict ```demand for bike usage```.    
Add SMOTE and feature expansion to reduce model bias and better fit the training dataset. Report precision, recall,    
and F1-score to compare model performance. The **regression model with feature expansion** outperforms the rest.

---

**Observation:-**
Logistic regression with feature expansion has the best metrics compared to the other two methods.
| Model | F1-score on validation set |
| :- | :- |
| Logistic Regression | 0.85 |
| Regression with SMOTE | 0.82 |
| Regression with feature expansion | 0.86 |
