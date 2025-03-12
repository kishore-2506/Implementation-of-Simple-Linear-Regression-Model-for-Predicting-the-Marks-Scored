# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kishore M
RegisterNumber:  212223040100
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

```

```
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

```
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

### DATASET:
![Dataset](https://github.com/user-attachments/assets/56c927b2-c11d-4369-9e0b-9495a7bc7236)

### HEAD VALUES DISPLAY:
![image](https://github.com/user-attachments/assets/239d324f-d10d-42d1-aa6c-afa7958aadf6)

### TAIL VALUES DISPLAY:
![Tail Values Display](https://github.com/user-attachments/assets/a629e26f-7bef-4f4d-881e-749ecbd37011)

### X AND Y VALUES:
![X and Y Values](https://github.com/user-attachments/assets/8d3463c3-7c9f-43b6-97d2-0be68e44e0a5)

### X AND Y PREDICTION VALUES:
![Prediction values of X and Y](https://github.com/user-attachments/assets/afbbfd5a-bd17-4199-9da1-4397fbd05f5f)

### TRAINING SET:
![Training Set](https://github.com/user-attachments/assets/f77be0b7-7108-4c16-b6f3-ae4ecfc44ba1)

### TESTING SET:
![image](https://github.com/user-attachments/assets/81b65738-bb06-445c-8b9a-293ca737aba4)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
