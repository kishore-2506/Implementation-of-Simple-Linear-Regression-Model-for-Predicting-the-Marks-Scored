# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Magesh C M
RegisterNumber:  212223220053
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_path = r"C:\Users\admin\Desktop\student_scores.csv" 
data = pd.read_csv(file_path)


X = np.array(data['Hours'])
Y = np.array(data['Scores'])

Xmean = np.mean(X)
Ymean = np.mean(Y)

num, den = 0, 0
for i in range(len(X)):
    num += (X[i] - Xmean) * (Y[i] - Ymean)
    den += (X[i] - Xmean) ** 2

m = num / den
c = Ymean - m * Xmean


print(f"Equation of Regression Line: Y = {m:.2f}X + {c:.2f}")

Y_pred = m * X + c
print("Predicted Marks:", Y_pred)


plt.scatter(X, Y, color="blue", label="Actual Data")
plt.plot(X, Y_pred, color="red", label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks Scored")
plt.title("Study Hours vs Marks (Linear Regression)")
plt.legend()
plt.show()


```

## Output:
![image](https://github.com/user-attachments/assets/3b9c884a-139b-4497-b62d-755eaa1b4e05)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
