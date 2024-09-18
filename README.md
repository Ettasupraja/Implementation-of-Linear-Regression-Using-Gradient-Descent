# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

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
Program to implement the linear regression using gradient descent.
Developed by: ETTA SUPRAJA
RegisterNumber:  212223220022
*/
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

  Hours  Scores
0    2.5      21
1    5.1      47
2    3.2      27
3    8.5      75
4    3.5      30
    Hours  Scores
20    2.7      30
21    4.8      54
22    3.8      35
23    6.9      76
24    7.8      86

x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)


[[2.5]
 [5.1]
 [3.2]
 [8.5]
 [3.5]
 [1.5]
 [9.2]
 [5.5]
 [8.3]
 [2.7]
 [7.7]
 [5.9]
 [4.5]
 [3.3]
 [1.1]
 [8.9]
 [2.5]
 [1.9]
 [6.1]
 [7.4]
 [2.7]
 [4.8]
 [3.8]
 [6.9]
 [7.8]]
[21 47 27 75 30 20 88 60 81 25 85 62 41 42 17 95 30 24 67 69 30 54 35 76
 86]

X.shape

(25, 1)

Y.shape

(25,)

m=0
c=0
L=0.001 # learning rate
epochs=5000 # No.of iterations to be performed
n=float(len(x))
error=[]
# Performing Gradient Descent
for i in range(epochs):
  y_pred = m*x + c
  D_m = (-2/n)*sum(x*(y-y_pred))
  D_c = (-2/n)*sum(y-y_pred)
  m = m-L*D_m
  c = c-L*D_c
  error.append(sum(y-y_pred)**2)
print(m,c)
type(error)
print(len(error))
plt.plot(range(0,epochs),error)

[0.48911015 1.0946751  0.62885591 1.74682196 0.69872879 0.46581919
 2.04960444 1.39745757 1.88656772 0.58227399 1.97973156 1.44403949
 0.95492934 0.9782203  0.39594631 2.21264115 0.69872879 0.55898303
 1.56049429 1.60707621 0.69872879 1.25771181 0.81518358 1.77011292
 2.00302252] [17.96987707 40.21829629 23.10412766 64.17813238 25.67125295 17.11416864
 75.30234199 51.34250591 69.31238297 21.39271079 72.7352167  53.05392277
 35.0840457  35.93975413 14.54704334 81.29230102 25.67125295 20.53700236
 57.33246493 59.04388179 25.67125295 46.20825532 29.94979511 65.03384081
 73.59092513]
```
## Output:
![image](https://github.com/user-attachments/assets/143288b0-fa54-4b07-82d5-922d75e6514b)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
