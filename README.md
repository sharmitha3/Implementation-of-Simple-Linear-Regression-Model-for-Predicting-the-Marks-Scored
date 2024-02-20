# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas ,numpy and required functions from sklearn.
2. Read the csv file using read_csv.
3. Separate the independent variable from the dependent variable.
4. Split the data.
5. Create a regression model.
6. Calculate the MSE,MAE,RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SHARMITHA V
RegisterNumber:  212223110048
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="orange")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="red")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:
![image](https://github.com/sharmitha3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145974496/8e50f573-b3e9-489f-93dd-ae488ff9aca5)
![image](https://github.com/sharmitha3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145974496/0611e920-9f04-4059-9d76-99ffe59d8a62)
![image](https://github.com/sharmitha3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145974496/a36428a2-66e1-400e-9f53-223d1c94e89c)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
