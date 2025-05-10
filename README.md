# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import Libraries: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.
2.Load Dataset: Import the dataset containing car prices along with relevant features.
3.Data Preprocessing: Manage missing data and select key features for the model, if required.
4.Split Data: Divide the dataset into training and testing subsets.
5.Train Model: Build a linear regression model and train it using the training data.
6.Make Predictions: Apply the model to predict outcomes for the test set.
7.Evaluate Model: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8.Check Assumptions: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.
9.Output Results: Present the predictions and evaluation metrics.
```

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: G.Ramanujam
RegisterNumber:  212224240129

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment.csv')
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
print("="*50)
print("MODEL COEFFICIENTS:")
for feature,coef in zip(x.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}:{model.intercept_:>10.2f}")
print("\n MODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test,y_pred):>10.2f}")
print(f"{'RMSE':>12}:{np.sqrt(mean_squared_error(y_test,y_pred)):>10.2f}")
print(f"{'R-squares':>12}:{r2_score(y_test,y_pred):10.2f}")
print("="*50)
# 1. Linearity check
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

# 2. Independence (Durbin-Watson)
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicate no autocorrelation)")

# 3. Homoscedasticity
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
# 4. Normality of residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
*/

```

## Output:

## MODEL COEFFICIENTS AND MODEL PERFORMANCE:
![Screenshot 2025-05-10 113404](https://github.com/user-attachments/assets/e646bb41-9746-4eec-980d-56256569b780)
## LINEARITY CHECK:
![Screenshot 2025-05-10 113423](https://github.com/user-attachments/assets/d5956053-bcb3-4be1-8e55-ae6b04f18f97)
## INDEPENDENCE (DURBIN-WATSON):
![Screenshot 2025-05-10 113430](https://github.com/user-attachments/assets/19e34f82-1a54-4d52-b7b4-5d508c292517)
## HOMOSCEDASTICITY:
![Screenshot 2025-05-10 113438](https://github.com/user-attachments/assets/d0df8b53-7eb0-415e-bfdc-52e7e5ad5684)
## NORMALITY OF RESIDUALS:
![Screenshot 2025-05-10 113447](https://github.com/user-attachments/assets/a7dd4f0f-c7a5-40e4-8e55-4d82413e50cf)


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
